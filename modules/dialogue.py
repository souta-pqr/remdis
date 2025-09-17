import sys
import threading
import queue
import time
import re

from base import RemdisModule, RemdisState, RemdisUtil, RemdisUpdateType
import prompt.util as prompt_util

# RAGヘルパーのインポート
try:
    from rag_helper import RAGHelper, load_rag_config
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: RAGヘルパーのインポートに失敗: {e}")
    RAG_AVAILABLE = False

# 既存のLLMインポート
from llm import ResponseChatGPT


class Dialogue(RemdisModule):
    def __init__(self, 
                 pub_exchanges=['dialogue', 'dialogue2'],
                 sub_exchanges=['asr', 'vap', 'tts', 'bc', 'emo_act']):
        super().__init__(pub_exchanges=pub_exchanges,
                         sub_exchanges=sub_exchanges)

        # 設定の読み込み
        self.history_length = self.config['DIALOGUE']['history_length']
        self.response_generation_interval = self.config['DIALOGUE']['response_generation_interval']
        self.prompts = prompt_util.load_prompts(self.config['ChatGPT']['prompts'])

        # RAGヘルパーの初期化
        self.rag_helper = None
        if RAG_AVAILABLE:
            try:
                # RAG設定を含む完全な設定を読み込み
                full_config = load_rag_config()
                # 既存の設定とマージ
                for key, value in full_config.items():
                    if key not in self.config:
                        self.config[key] = value
                    elif isinstance(self.config[key], dict) and isinstance(value, dict):
                        self.config[key].update(value)
                
                self.rag_helper = RAGHelper(self.config, self.prompts)
                print("RAGヘルパーを初期化しました")
            except Exception as e:
                print(f"RAGヘルパー初期化エラー: {e}")
                self.rag_helper = None
        
        # 対話履歴
        self.dialogue_history = []

        # IUおよび応答の処理用バッファ
        self.system_utterance_end_time = 0.0
        self.input_iu_buffer = queue.Queue()
        self.bc_iu_buffer = queue.Queue()
        self.emo_act_iu_buffer = queue.Queue()
        self.output_iu_buffer = []
        self.llm_buffer = queue.Queue()

        # 対話状態管理
        self.event_queue = queue.Queue()
        self.state = 'idle'
        self._is_running = True

        # IU処理用の関数
        self.util_func = RemdisUtil()

        # 応答生成管理用の変数
        self.pending_llm_tasks = {}  # タスクIDと関連LLMを管理
        self.llm_task_counter = 0

    # メインループ
    def run(self):
        self.log("***** DIALOGUE MODULE STARTING *****")
        self.log(f"***** Initial state: {self.state} *****")
        
        # 音声認識結果受信スレッド
        t1 = threading.Thread(target=self.listen_asr_loop)
        # 音声合成結果受信スレッド
        t2 = threading.Thread(target=self.listen_tts_loop)
        # ターンテイキングイベント受信スレッド
        t3 = threading.Thread(target=self.listen_vap_loop)
        # 相槌生成結果受信スレッド
        t4 = threading.Thread(target=self.listen_bc_loop)
        # 表情・行動情報受信スレッド
        t5 = threading.Thread(target=self.listen_emo_act_loop)
        # 逐次応答生成スレッド
        t6 = threading.Thread(target=self.parallel_response_generation)
        # 状態制御スレッド
        t7 = threading.Thread(target=self.state_management)
        # 表情・行動制御スレッド
        t8 = threading.Thread(target=self.emo_act_management)

        # スレッド実行
        t1.start()
        t2.start()
        t3.start()
        t4.start()
        t5.start()
        t6.start()
        t7.start()
        t8.start()

    # 音声認識結果受信用のコールバックを登録
    def listen_asr_loop(self):
        self.log("***** SUBSCRIBING TO ASR CHANNEL *****")
        self.subscribe('asr', self.callback_asr)

    # 音声合成結果受信用のコールバックを登録
    def listen_tts_loop(self):
        self.log("***** SUBSCRIBING TO TTS CHANNEL *****")
        self.subscribe('tts', self.callback_tts)

    # VAP情報受信用のコールバックを登録
    def listen_vap_loop(self):
        self.log("***** SUBSCRIBING TO VAP CHANNEL *****")
        self.subscribe('vap', self.callback_vap)

    # バックチャネル受信用のコールバックを登録
    def listen_bc_loop(self):
        self.log("***** SUBSCRIBING TO BC CHANNEL *****")
        self.subscribe('bc', self.callback_bc)

    # 表情・行動情報受信用のコールバックを登録
    def listen_emo_act_loop(self):
        self.log("***** SUBSCRIBING TO EMO_ACT CHANNEL *****")
        self.subscribe('emo_act', self.callback_emo_act)

    # 随時受信される音声認識結果に対して並列に応答を生成
    def parallel_response_generation(self):
        self.log("***** PARALLEL RESPONSE GENERATION STARTED *****")
        # 受信したIUを保持しておく変数
        iu_memory = []
        new_iu_count = 0

        while True:
            # IUを受信して保存
            input_iu = self.input_iu_buffer.get()
            iu_memory.append(input_iu)
            
            # IUがREVOKEだった場合はメモリから削除
            if input_iu['update_type'] == RemdisUpdateType.REVOKE:
                iu_memory = self.util_func.remove_revoked_ius(iu_memory)
            # ADD/COMMITの場合は応答候補生成
            else:
                user_utterance = self.util_func.concat_ius_body(iu_memory)
                if user_utterance == '':
                    continue

                # ADDの場合は閾値以上のIUが溜まっているか確認し，溜まっていなければ次のIUもしくはCOMMITを待つ
                if input_iu['update_type'] == RemdisUpdateType.ADD:
                    new_iu_count += 1
                    if new_iu_count < self.response_generation_interval:
                        continue
                    else:
                        new_iu_count = 0

                # パラレルな応答生成処理（RAG統合）
                self.log(f"***** CREATING LLM FOR USER UTTERANCE: '{user_utterance}' *****")
                
                # タスクID生成
                task_id = self.llm_task_counter
                self.llm_task_counter += 1
                
                # LLMを作成してタスクを開始
                llm = self._create_response_llm()
                last_asr_iu_id = input_iu['id']
                
                # タスクを記録
                self.pending_llm_tasks[task_id] = {
                    'llm': llm,
                    'timestamp': input_iu['timestamp'],
                    'user_utterance': user_utterance,
                    'asr_iu_id': last_asr_iu_id
                }
                
                # 非同期でLLM実行
                t = threading.Thread(
                    target=self._run_llm_with_completion,
                    args=(task_id, llm, input_iu['timestamp'], user_utterance, 
                          self.dialogue_history, last_asr_iu_id)
                )
                t.start()

                # ユーザ発話終端の処理
                if input_iu['update_type'] == RemdisUpdateType.COMMIT:
                    iu_memory = []

    def _run_llm_with_completion(self, task_id, llm, asr_timestamp, user_utterance, 
                               dialogue_history, last_asr_iu_id):
        """LLMを実行し、完了後にバッファに追加"""
        try:
            self.log(f"***** STARTING LLM TASK {task_id} *****")
            
            # LLM実行
            llm.run(asr_timestamp, user_utterance, dialogue_history, last_asr_iu_id, queue.Queue())
            
            # LLMバッファに追加
            self.llm_buffer.put(llm)
            self.log(f"***** LLM TASK {task_id} COMPLETED AND ADDED TO BUFFER *****")
            
        except Exception as e:
            self.log(f"***** LLM TASK {task_id} ERROR: {e} *****")
            import traceback
            traceback.print_exc()
        finally:
            # タスクをクリア
            if task_id in self.pending_llm_tasks:
                del self.pending_llm_tasks[task_id]

    def _create_response_llm(self):
        """適切なLLMを作成（RAG機能統合）"""
        # RAGヘルパーが利用可能な場合
        if self.rag_helper:
            try:
                self.log("***** CREATING RAG-ENABLED LLM *****")
                return self.rag_helper.create_response_llm(self.config, self.prompts)
            except Exception as e:
                self.log(f"RAG LLM作成エラー: {e}")
                # フォールバックに進む
        
        # 通常のLLMを使用（元のdialogue.pyと同じ動作）
        self.log("***** CREATING STANDARD LLM *****")
        return ResponseChatGPT(self.config, self.prompts)

    # 対話状態を管理
    def state_management(self):
        self.log("***** STATE MANAGEMENT STARTED *****")
        while True:
            # イベントに応じて状態を遷移
            event = self.event_queue.get()
            prev_state = self.state
            
            self.log(f"***** RECEIVED EVENT: {event} in state {prev_state} *****")
            
            # 状態遷移を実行
            if event in RemdisState.transition[self.state]:
                self.state = RemdisState.transition[self.state][event]
                self.log(f'********** State: {prev_state} -> {self.state}, Trigger: {event} **********')
            else:
                self.log(f"***** WARNING: Invalid event {event} for state {prev_state} *****")
                continue
            
            # 直前の状態がtalkingの場合にイベントに応じて処理を実行
            if prev_state == 'talking':
                if event == 'SYSTEM_BACKCHANNEL':
                    self.log("***** HANDLING SYSTEM_BACKCHANNEL IN TALKING STATE *****")
                    pass
                if event == 'USER_BACKCHANNEL':
                    self.log("***** HANDLING USER_BACKCHANNEL IN TALKING STATE *****")
                    pass
                if event == 'USER_TAKE_TURN':
                    self.log("***** HANDLING USER_TAKE_TURN - STOPPING RESPONSE *****")
                    self.stop_response()
                if event == 'BOTH_TAKE_TURN':
                    self.log("***** HANDLING BOTH_TAKE_TURN - STOPPING RESPONSE *****")
                    self.stop_response()
                if event == 'TTS_COMMIT':
                    self.log("***** HANDLING TTS_COMMIT - STOPPING RESPONSE *****")
                    self.stop_response()
                
            # 直前の状態がidleの場合にイベントに応じて処理を実行
            elif prev_state == 'idle':
                if event == 'SYSTEM_BACKCHANNEL':
                    self.log("***** HANDLING SYSTEM_BACKCHANNEL - SENDING BACKCHANNEL *****")
                    self.send_backchannel()
                if event == 'SYSTEM_TAKE_TURN':
                    self.log("***** HANDLING SYSTEM_TAKE_TURN - SENDING RESPONSE *****")
                    self.send_response()

    # 表情・感情を管理
    def emo_act_management(self):
        self.log("***** EMO_ACT MANAGEMENT STARTED *****")
        while True:
            iu = self.emo_act_iu_buffer.get()
            self.log(f"***** RECEIVED EMO_ACT: {iu['body']} *****")
            # 感情または行動の送信
            expression_and_action = {}
            if 'emotion' in iu['body']:
                expression_and_action['expression'] = iu['body']['emotion']
            if 'action' in iu['body']:
                expression_and_action['action'] = iu['body']['action']
            
            if expression_and_action:
                snd_iu = self.createIU(expression_and_action, 'dialogue2', RemdisUpdateType.ADD)
                snd_iu['data_type'] = 'expression_and_action'
                self.log(f"***** SENDING EXPRESSION_AND_ACTION: {expression_and_action} *****")
                self.printIU(snd_iu)
                self.publish(snd_iu, 'dialogue2')

    # システム発話を送信（SYSTEM_TAKE_TURNのみで発話）
    def send_response(self):
        self.log(f"***** SEND_RESPONSE CALLED - Current state: {self.state} *****")
        self.log(f"***** LLM buffer empty: {self.llm_buffer.empty()} *****")
        
        # LLMバッファが空の場合、少し長めに待機
        max_wait_time = 2.0  # 最大2秒待機
        wait_interval = 0.1  # 0.1秒間隔でチェック
        waited_time = 0.0
        
        while self.llm_buffer.empty() and waited_time < max_wait_time:
            self.log(f"***** WAITING FOR LLM... ({waited_time:.1f}s) *****")
            time.sleep(wait_interval)
            waited_time += wait_interval
        
        if self.llm_buffer.empty():
            self.log("***** LLM BUFFER STILL EMPTY - CREATING NEW LLM *****")
            # 新しいLLMを作成
            llm = self._create_response_llm()
            t = threading.Thread(
                target=self._run_llm_with_completion,
                args=(-1, llm, time.time(), None, self.dialogue_history, None)
            )
            t.start()
            
            # 追加で少し待機
            time.sleep(1.0)

        # 応答が生成され始めたLLMの中で一番新しい音声認識結果を使っているものを選択して送信
        if not self.llm_buffer.empty():
            self.log("***** SELECTING LLM FROM BUFFER *****")
            selected_llm = self.llm_buffer.get()
            
            # バッファに複数のLLMがある場合、最新のものを選択
            latest_asr_time = getattr(selected_llm, 'asr_time', 0)
            while not self.llm_buffer.empty():
                llm = self.llm_buffer.get()
                llm_asr_time = getattr(llm, 'asr_time', 0)
                if llm_asr_time > latest_asr_time:
                    selected_llm = llm
                    latest_asr_time = llm_asr_time

            # IUに分割して送信
            user_utterance = getattr(selected_llm, 'user_utterance', '(不明)')
            self.log(f'***** SELECTED USER UTTERANCE: {user_utterance} *****')
            
            # 応答イテレータを取得
            response_iterator = getattr(selected_llm, 'response', None)
            if response_iterator is not None:
                conc_response = ''
                response_count = 0
                
                self.log("***** STARTING RESPONSE GENERATION *****")
                try:
                    for part in response_iterator:
                        response_count += 1
                        self.log(f"***** RESPONSE PART {response_count}: {part} *****")
                        
                        # 表情・動作を送信
                        expression_and_action = {}
                        if 'expression' in part and part['expression'] != 'normal':
                            expression_and_action['expression'] = part['expression']
                        if 'action' in part and part['action'] != 'wait':
                            expression_and_action['action'] = part['action']
                        if expression_and_action:
                            snd_iu = self.createIU(expression_and_action, 'dialogue2', RemdisUpdateType.ADD)
                            snd_iu['data_type'] = 'expression_and_action'
                            self.log(f"***** SENDING EXPRESSION_AND_ACTION: {expression_and_action} *****")
                            self.printIU(snd_iu)
                            self.publish(snd_iu, 'dialogue2')
                            self.output_iu_buffer.append(snd_iu)

                        # 生成中に状態が変わることがあるためその確認の後，発話を送信
                        if 'phrase' in part:
                            phrase = part['phrase']
                            self.log(f"***** CURRENT STATE: {self.state}, PHRASE: '{phrase}' *****")
                            if self.state == 'talking':
                                snd_iu = self.createIU(phrase, 'dialogue', RemdisUpdateType.ADD)
                                self.log(f"***** SENDING PHRASE TO TTS: '{phrase}' *****")
                                self.printIU(snd_iu)
                                self.publish(snd_iu, 'dialogue')
                                self.output_iu_buffer.append(snd_iu)
                                conc_response += phrase
                            else:
                                self.log(f"***** NOT SENDING PHRASE - STATE IS {self.state} *****")
                                
                        # 無限ループ防止
                        if response_count > 50:
                            self.log("***** TOO MANY RESPONSE PARTS - BREAKING *****")
                            break
                            
                except Exception as e:
                    self.log(f"***** RESPONSE ITERATION ERROR: {e} *****")
                    import traceback
                    traceback.print_exc()

                # 対話コンテキストにユーザ発話を追加
                if user_utterance and user_utterance != '(不明)':
                    self.history_management('user', user_utterance)
                else:
                    self.history_management('user', '(沈黙)')
                self.history_management('assistant', conc_response)

                self.log(f"***** TOTAL RESPONSE: '{conc_response}' *****")
            else:
                self.log("***** NO RESPONSE ITERATOR AVAILABLE *****")

            # 応答生成終了メッセージ
            self.log('***** SENDING DIALOGUE COMMIT *****')
            snd_iu = self.createIU('', 'dialogue', RemdisUpdateType.COMMIT)
            self.printIU(snd_iu)
            self.publish(snd_iu, 'dialogue')
        else:
            self.log("***** WARNING: NO LLM AVAILABLE IN BUFFER AFTER WAITING *****")

    # バックチャネルを送信
    def send_backchannel(self):
        self.log("***** SEND_BACKCHANNEL CALLED *****")
        if not self.bc_iu_buffer.empty():
            iu = self.bc_iu_buffer.get()

            # 現在の状態がidleの場合のみ後続の処理を実行してバックチャネルを送信
            if self.state != 'idle':
                self.log(f"***** NOT SENDING BACKCHANNEL - STATE IS {self.state} *****")
                return

            # 相槌の送信
            self.log(f"***** SENDING BACKCHANNEL: {iu['body']['bc']} *****")
            snd_iu = self.createIU(iu['body']['bc'], 'dialogue', RemdisUpdateType.ADD)
            self.printIU(snd_iu)
            self.publish(snd_iu, 'dialogue')
        else:
            self.log("***** NO BACKCHANNEL IN BUFFER *****")

    # 応答を中断
    def stop_response(self):
        self.log("***** STOPPING RESPONSE *****")
        for iu in self.output_iu_buffer:
            iu['update_type'] = RemdisUpdateType.REVOKE
            self.log(f"***** REVOKING IU: {iu} *****")
            self.printIU(iu)
            self.publish(iu, iu['exchange'])
        self.output_iu_buffer = []

    # 音声認識結果受信用のコールバック
    def callback_asr(self, ch, method, properties, in_msg):
        in_msg = self.parse_msg(in_msg)
        self.log(f"***** RECEIVED ASR: {in_msg['body']} ({in_msg['update_type']}) *****")
        self.input_iu_buffer.put(in_msg)
            
    # 音声合成結果受信用のコールバック
    def callback_tts(self, ch, method, properties, in_msg):
        in_msg = self.parse_msg(in_msg)
        self.log(f"***** RECEIVED TTS: {in_msg['update_type']} *****")
        if in_msg['update_type'] == RemdisUpdateType.COMMIT:
            self.output_iu_buffer = []
            self.system_utterance_end_time = in_msg['timestamp']
            self.log("***** SENDING TTS_COMMIT EVENT *****")
            self.event_queue.put('TTS_COMMIT')

    # VAP情報受信用のコールバック
    def callback_vap(self, ch, method, properties, in_msg):
        in_msg = self.parse_msg(in_msg)
        event = in_msg['body']
        self.log(f"***** RECEIVED VAP EVENT: {event} *****")
        self.event_queue.put(event)

    # バックチャネル受信用のコールバック
    def callback_bc(self, ch, method, properties, in_msg):
        in_msg = self.parse_msg(in_msg)
        self.log(f"***** RECEIVED BC: {in_msg['body']} *****")
        self.bc_iu_buffer.put(in_msg)
        self.log("***** SENDING SYSTEM_BACKCHANNEL EVENT *****")
        self.event_queue.put('SYSTEM_BACKCHANNEL')

    # 表情・行動情報受信用のコールバック
    def callback_emo_act(self, ch, method, properties, in_msg):
        in_msg = self.parse_msg(in_msg)
        self.log(f"***** RECEIVED EMO_ACT: {in_msg['body']} *****")
        self.emo_act_iu_buffer.put(in_msg)

    # 対話履歴を更新
    def history_management(self, role, utt):
        self.dialogue_history.append({"role": role, "content": utt})
        if len(self.dialogue_history) > self.history_length:
            self.dialogue_history.pop(0)

    # RAG統計情報を取得（デバッグ用）
    def get_rag_stats(self):
        """RAG統計情報を取得"""
        if self.rag_helper:
            return self.rag_helper.get_rag_stats()
        else:
            return {
                'rag_enabled': False,
                'status': 'not_available'
            }

    # デバッグ用にログを出力
    def log(self, *args, **kwargs):
        print(f"[DIALOGUE-{time.time():.5f}]", *args, flush=True, **kwargs)

def main():
    dialogue = Dialogue()
    dialogue.run()

if __name__ == '__main__':
    main()
    