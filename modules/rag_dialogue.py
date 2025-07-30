import sys
import threading
import queue
import time
import re
import os


from base import RemdisModule, RemdisUpdateType


try:
    from rag_retriever import RAGRetriever
except ImportError:
    class RAGRetriever:
        def __init__(self, config):
            self.config = config
            # 藤江研究室に関する基本的な知識ベース
            self.knowledge_base = {
                '研究室について': '藤江研究室は音声対話技術の研究を行っています。特に自然な音声対話システムの開発に取り組んでいます。',
                'メンバー': '研究室には教授、学生、研究員が所属しています。',
                '研究分野': '音声認識、自然言語処理、対話システム、マルチモーダル対話などの研究を行っています。',
                '技術': 'Remdisフレームワークを使用した音声対話システムの開発を行っています。',
                '挨拶': 'こんにちは！藤江研究室へようこそ。音声対話技術について何かご質問がありましたらお聞かせください。'
            }
            
            # 収集されたウェブデータ用のストレージ
            self.web_documents = []
            
        def retrieve(self, query):
            if not query or query.strip() == '':
                return {"level": 3, "type": "general", "content": "質問が空です", "confidence": 0.1}
            
            query_lower = query.lower()
            
            # まず収集されたウェブデータから検索
            web_result = self._search_web_documents(query_lower)
            if web_result:
                return web_result
            
            # 基本知識ベースから検索
            return self._search_basic_knowledge(query_lower)
        
        def _search_web_documents(self, query_lower):
            """収集されたウェブ文書から検索"""
            if not self.web_documents:
                return None
            
            best_match = None
            best_score = 0.0
            
            for doc in self.web_documents:
                content = doc.get('content', '').lower()
                title = doc.get('title', '').lower()
                
                # 簡単なキーワードマッチング
                score = 0.0
                query_words = query_lower.split()
                
                for word in query_words:
                    if word in content:
                        score += 1.0
                    if word in title:
                        score += 2.0  # タイトルマッチは高得点
                
                if score > best_score:
                    best_score = score
                    best_match = doc
            
            if best_match and best_score > 0:
                return {
                    "level": 1,
                    "type": "website",
                    "content": best_match.get('content', '')[:500] + '...',  # 最初の500文字
                    "confidence": min(0.9, best_score / len(query_lower.split())),
                    "source": best_match.get('source', ''),
                    "title": best_match.get('title', '')
                }
            
            return None
        
        def _search_basic_knowledge(self, query_lower):
            """基本知識ベースから検索"""
            # キーワードベースのマッチング
            if any(word in query_lower for word in ['こんにちは', 'hello', '挨拶']):
                return {"level": 1, "type": "greeting", "content": self.knowledge_base['挨拶'], "confidence": 0.9}
            elif any(word in query_lower for word in ['研究室', 'ラボ', 'lab']):
                return {"level": 1, "type": "about", "content": self.knowledge_base['研究室について'], "confidence": 0.8}
            elif any(word in query_lower for word in ['メンバー', 'スタッフ', '人', '誰']):
                return {"level": 2, "type": "members", "content": self.knowledge_base['メンバー'], "confidence": 0.7}
            elif any(word in query_lower for word in ['研究', '技術', 'research']):
                return {"level": 1, "type": "research", "content": self.knowledge_base['研究分野'], "confidence": 0.8}
            elif any(word in query_lower for word in ['remdis', 'システム', 'フレームワーク']):
                return {"level": 1, "type": "technology", "content": self.knowledge_base['技術'], "confidence": 0.8}
            else:
                # デフォルトの応答
                return {"level": 2, "type": "general", "content": f"「{query_lower}」について詳しい情報をお探しですね。藤江研究室では音声対話技術の研究を行っており、様々なトピックについてお答えできるかもしれません。より具体的にお聞かせください。", "confidence": 0.5}
        
        def add_documents(self, documents):
            """文書を追加"""
            if not documents:
                return False
            
            print(f"知識ベースに{len(documents)}件の文書を追加しています...")
            
            for doc in documents:
                if isinstance(doc, dict) and doc.get('content'):
                    self.web_documents.append(doc)
            
            print(f"追加完了: 合計{len(self.web_documents)}件の文書が利用可能です")
            return True
        
        def get_collection_info(self):
            """コレクション情報を返す"""
            return {
                'status': 'active',
                'count': len(self.knowledge_base) + len(self.web_documents),
                'web_documents': len(self.web_documents),
                'basic_knowledge': len(self.knowledge_base)
            }




# --- RemdisModule継承でtin/toutと連携できるRAG対話モジュール ---
class RAGDialogue(RemdisModule):
    """RAG機能付きRemdis対話モジュール（asr購読→dialogue出力）"""
    def __init__(self, pub_exchanges=['dialogue'], sub_exchanges=['asr'], **kwargs):
        super().__init__(pub_exchanges=pub_exchanges, sub_exchanges=sub_exchanges)
        config = kwargs.pop('config', None)
        self.config = config if config is not None else {
            'DIALOGUE': {'history_length': 10, 'response_generation_interval': 2},
            'RAG': {'enable': True},
            'ChatGPT': {'prompts': {}}
        }
        self.rag_retriever = RAGRetriever(self.config)
        
        # RAG統計情報の初期化
        self.rag_stats = {
            'queries_processed': 0,
            'level_1_responses': 0,
            'level_2_responses': 0,
            'level_3_responses': 0,
            'average_response_time': 0.0,
            'knowledge_base_size': 0
        }
        
        # ユーザー発話の蓄積用バッファ
        self.user_utterance_buffer = []
        
        # RAG機能の設定
        self.rag_enabled = self.config.get('RAG', {}).get('enable', True)
        
        # データ収集器の初期化
        try:
            from data_collector import FujielabDataCollector
            self.data_collector = FujielabDataCollector(self.config)
            print("データ収集器を初期化しました")
        except ImportError as e:
            print(f"データ収集器のインポートに失敗: {e}")
            self.data_collector = None
        
        # 基本的な対話機能で必要な属性（フォールバック用）
        try:
            # プロンプト設定ファイルからの読み込み
            from prompt.util import load_prompts
            prompt_dict = self.config.get('ChatGPT', {}).get('prompts', {})
            if prompt_dict:
                self.prompts = load_prompts(prompt_dict)
            else:
                self.prompts = {'RESP': 'あなたは藤江研究室の案内AIです。', 'TO': '何かご質問はありますか？'}
        except ImportError:
            self.prompts = {'RESP': 'あなたは藤江研究室の案内AIです。', 'TO': '何かご質問はありますか？'}
        
        self.dialogue_history = []
        self.history_length = self.config.get('DIALOGUE', {}).get('history_length', 10)
        self.response_generation_interval = self.config.get('DIALOGUE', {}).get('response_generation_interval', 2)
        
        # バッファ類
        self.input_iu_buffer = queue.Queue()
        self.bc_iu_buffer = queue.Queue()
        self.emo_act_iu_buffer = queue.Queue()
        self.output_iu_buffer = []
        self.llm_buffer = queue.Queue()
        
        # 状態管理
        self.event_queue = queue.Queue()
        self.state = 'idle'
        self.system_utterance_end_time = 0.0
        
        # ユーティリティ
        self.util_func = type('RemdisUtil', (), {
            'remove_revoked_ius': lambda self, ius: [iu for iu in ius if iu.get('update_type') != 'revoke'],
            'concat_ius_body': lambda self, ius: ''.join([iu.get('body', '') for iu in ius])
        })()
        
        # RAGコンポーネントの初期化
        if self.rag_enabled:
            self._initialize_rag_components()
        
        self._is_running = True
        print("RAG対話Remdisモジュールを初期化しました")

    def callback_asr(self, ch, method, properties, in_msg):
        iu = self.parse_msg(in_msg)
        print('IN(asr):', end='')
        self.printIU(iu, flush=True)
        
        if iu['update_type'] == RemdisUpdateType.ADD:
            # ADDメッセージの場合、発話内容をバッファに追加
            user_text = iu.get('body', '')
            if user_text.strip():  # 空でない場合のみ追加
                self.user_utterance_buffer.append(user_text)
                
        elif iu['update_type'] == RemdisUpdateType.COMMIT:
            # COMMITメッセージの場合、蓄積された発話を処理
            if self.user_utterance_buffer:
                # バッファの内容を結合
                user_utt = ''.join(self.user_utterance_buffer).strip()
                print(f"処理する発話: '{user_utt}'")
                
                if user_utt:
                    # RAG検索実行
                    start_time = time.time()
                    rag_result = self.rag_retriever.retrieve(user_utt)
                    response_time = time.time() - start_time
                    
                    # 統計更新
                    self._update_rag_stats(rag_result, response_time)
                    
                    # 応答内容の決定
                    if isinstance(rag_result, dict):
                        resp = rag_result.get('content', 'すみません、適切な回答が見つかりませんでした。')
                        if not resp or resp.strip() == '':
                            resp = f"「{user_utt}」についてお答えします。藤江研究室は音声対話技術の研究を行っています。"
                    else:
                        resp = str(rag_result)
                    
                    # 応答送信
                    snd_iu = self.createIU(resp, 'dialogue', RemdisUpdateType.ADD)
                    print('OUT(dialogue):', end='')
                    self.printIU(snd_iu, flush=True)
                    self.publish(snd_iu, 'dialogue')
                else:
                    # 空の発話の場合
                    resp = "何かご質問はありますか？"
                    snd_iu = self.createIU(resp, 'dialogue', RemdisUpdateType.ADD)
                    print('OUT(dialogue):', end='')
                    self.printIU(snd_iu, flush=True)
                    self.publish(snd_iu, 'dialogue')
                
                # バッファクリア
                self.user_utterance_buffer = []
            else:
                # バッファが空の場合
                resp = "何かご質問はありますか？"
                snd_iu = self.createIU(resp, 'dialogue', RemdisUpdateType.ADD)
                print('OUT(dialogue):', end='')
                self.printIU(snd_iu, flush=True)
                self.publish(snd_iu, 'dialogue')
                
        elif iu['update_type'] == RemdisUpdateType.REVOKE:
            # REVOKEメッセージの場合、バッファから該当部分を削除
            # 簡単な実装として、最後の要素を削除
            if self.user_utterance_buffer:
                removed = self.user_utterance_buffer.pop()
                print(f"発話を取り消しました: '{removed}'")
    
    def _initialize_fallback(self, pub_exchanges, sub_exchanges, kwargs):
        """フォールバック初期化（既存モジュールが利用できない場合）"""
        # 基本的な属性を設定
        self.pub_exchanges = pub_exchanges
        self.sub_exchanges = sub_exchanges
        
        # 設定の読み込み（簡略版）
        self.config = kwargs.get('config', {
            'DIALOGUE': {'history_length': 10, 'response_generation_interval': 2},
            'RAG': {'enable': True},
            'ChatGPT': {'prompts': {}}
        })
        
        # プロンプトの読み込み
        self.prompts = {'RESP': 'あなたは対話AIです。', 'TO': '会話を始めましょう。'}
        
        # 対話履歴
        self.dialogue_history = []
        
        # 設定値
        self.history_length = self.config.get('DIALOGUE', {}).get('history_length', 10)
        self.response_generation_interval = self.config.get('DIALOGUE', {}).get('response_generation_interval', 2)
        
        # バッファ
        self.input_iu_buffer = queue.Queue()
        self.bc_iu_buffer = queue.Queue()
        self.emo_act_iu_buffer = queue.Queue()
        self.output_iu_buffer = []
        self.llm_buffer = queue.Queue()
        
        # 状態管理
        self.event_queue = queue.Queue()
        self.state = 'idle'
        self.system_utterance_end_time = 0.0
        
        # ユーティリティ
        self.util_func = type('RemdisUtil', (), {
            'remove_revoked_ius': lambda self, ius: [iu for iu in ius if iu.get('update_type') != 'revoke'],
            'concat_ius_body': lambda self, ius: ''.join([iu.get('body', '') for iu in ius])
        })()
        
        self._is_running = True
    
    def _initialize_rag_components(self):
        """RAG関連コンポーネントの初期化"""
        try:
            # RAG機能の有効/無効チェック
            if not self.rag_enabled:
                print("RAG機能は無効化されています")
                self.rag_retriever = None
                self.data_collector = None
                return
            
            # RAG検索エンジン初期化（既に初期化済み）
            print("RAG検索エンジンが利用可能です")
            
            # 初期データ収集（バックグラウンドで実行）
            if self.data_collector:
                self._start_initial_data_collection()
            
            print("RAG機能を有効化しました")
            
        except Exception as e:
            print(f"RAG初期化エラー: {e}")
            self.rag_enabled = False
            self.rag_retriever = None
            self.data_collector = None
    
    def _start_initial_data_collection(self):
        """初期データ収集をバックグラウンドで開始"""
        def collect_initial_data():
            try:
                print("初期データ収集を開始...")
                print("藤江研究室のウェブサイトから情報を収集しています...")
                
                # ウェブサイトからデータを収集
                documents = self.data_collector.collect_website_data(max_pages=10)
                
                if documents and self.rag_retriever:
                    # ChromaDBに文書を追加
                    success = self.rag_retriever.add_documents(documents)
                    if success:
                        self.rag_stats['knowledge_base_size'] = len(documents)
                        print(f"初期データ収集完了: {len(documents)}件の文書を知識ベースに追加")
                        
                        # 収集した文書のサンプルを表示
                        print("収集した文書の例:")
                        for i, doc in enumerate(documents[:3]):  # 最初の3件のみ
                            title = doc.get('title', '無題')
                            source = doc.get('source', 'unknown')
                            content_preview = doc.get('content', '')[:100] + '...'
                            print(f"  {i+1}. {title} (from: {source})")
                            print(f"      {content_preview}")
                        
                        print("RAGシステムで藤江研究室のウェブサイト情報が利用可能になりました。")
                    else:
                        print("文書の追加に失敗しました。基本知識ベースを使用します。")
                else:
                    print("初期データ収集: 有効な文書が見つかりませんでした")
                    print("モック知識ベースを使用します")
                        
            except Exception as e:
                print(f"初期データ収集エラー: {e}")
                print("モック知識ベースを使用します")
        
        # バックグラウンドスレッドで実行
        thread = threading.Thread(target=collect_initial_data, daemon=True)
        thread.start()
    
    def run(self):
        """メインループ - RAG対話システムの実行"""
        print("RAGDialogue: asrを購読しdialogueへ応答をpublishします")
        
        # RemdisModuleの基本機能を使用（存在する場合）
        try:
            self.subscribe('asr', self.callback_asr)
        except Exception as e:
            print(f"ASR購読エラー: {e}")
        
        # 親クラスのスレッドを起動（存在する場合）
        if hasattr(super(), 'run'):
            try:
                # 親クラスのスレッド起動
                self._start_parent_threads()
            except:
                # フォールバック
                self._start_fallback_threads()
        else:
            # フォールバック用のスレッド起動
            self._start_fallback_threads()
        
        # RAG専用スレッドを追加
        self._start_rag_threads()
        
        print("RAG対話システムが開始されました")
        
        # メインループを継続実行
        try:
            while self._is_running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nRAGDialogue: 終了シグナルを受信しました")
            self._is_running = False
    
    def _start_parent_threads(self):
        """親クラスのスレッドを起動"""
        # 既存のスレッド起動ロジックを呼び出し
        threads = [
            threading.Thread(target=self.listen_asr_loop),
            threading.Thread(target=self.listen_tts_loop),
            threading.Thread(target=self.listen_vap_loop),
            threading.Thread(target=self.listen_bc_loop),
            threading.Thread(target=self.listen_emo_act_loop),
            threading.Thread(target=self.parallel_response_generation),
            threading.Thread(target=self.state_management),
            threading.Thread(target=self.emo_act_management),
        ]
        
        for thread in threads:
            thread.start()
    
    def _start_fallback_threads(self):
        """フォールバック用のスレッド起動"""
        threads = [
            threading.Thread(target=self._fallback_input_loop),
            threading.Thread(target=self._fallback_response_loop),
        ]
        
        for thread in threads:
            thread.daemon = False  # メインスレッドと同じライフサイクル
            thread.start()
    
    def _start_rag_threads(self):
        """RAG専用スレッドを起動"""
        rag_threads = [
            threading.Thread(target=self._rag_stats_monitor, daemon=True),
        ]
        
        for thread in rag_threads:
            thread.start()
    
    def send_response(self):
        """RAG機能付き応答送信（既存メソッドをオーバーライド）"""
        try:
            if hasattr(self, 'llm_buffer') and self.llm_buffer.empty():
                # 少し待機してそれでも応答生成が始まらなければシステムから発話開始
                time.sleep(0.1)
                if self.llm_buffer.empty():
                    # RAG対応のLLMを使用
                    if self.rag_enabled and self.rag_retriever:
                        llm = RAGResponseChatGPT(self.config, self.prompts, self.rag_retriever)
                    else:
                        # フォールバック: 通常のLLM
                        try:
                            from llm import ResponseChatGPT
                            llm = ResponseChatGPT(self.config, self.prompts)
                        except ImportError:
                            llm = self._create_minimal_llm()
                    
                    t = threading.Thread(
                        target=llm.run,
                        args=(time.time(), None, self.dialogue_history, None, self.llm_buffer)
                    )
                    t.start()
            
            # 親クラスの応答送信ロジックを実行（存在する場合）
            if hasattr(super(), 'send_response'):
                super().send_response()
            else:
                self._fallback_send_response()
                
        except Exception as e:
            print(f"応答送信エラー: {e}")
            self._fallback_send_response()
    
    def parallel_response_generation(self):
        """RAG機能付き並列応答生成（既存メソッドをオーバーライド）"""
        if hasattr(super(), 'parallel_response_generation'):
            # 基本的には既存の実装を使用し、必要に応じてRAG機能を追加
            try:
                super().parallel_response_generation()
            except:
                self._fallback_response_generation()
        else:
            self._fallback_response_generation()
    
    def _fallback_response_generation(self):
        """フォールバック用の応答生成"""
        iu_memory = []
        new_iu_count = 0
        
        while self._is_running:
            try:
                # IU受信待機（タイムアウト付き）
                try:
                    input_iu = self.input_iu_buffer.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                iu_memory.append(input_iu)
                
                # IUがREVOKEだった場合はメモリから削除
                if input_iu.get('update_type') == 'revoke':
                    iu_memory = self.util_func.remove_revoked_ius(iu_memory)
                else:
                    user_utterance = self.util_func.concat_ius_body(iu_memory)
                    if not user_utterance:
                        continue
                    
                    # ADDの場合は閾値チェック
                    if input_iu.get('update_type') == 'add':
                        new_iu_count += 1
                        if new_iu_count < self.response_generation_interval:
                            continue
                        else:
                            new_iu_count = 0
                    
                    # RAG応答生成
                    self._generate_rag_response(user_utterance, input_iu)
                    
                    # COMMITの場合は履歴リセット
                    if input_iu.get('update_type') == 'commit':
                        iu_memory = []
                        
            except Exception as e:
                print(f"応答生成エラー: {e}")
                time.sleep(0.1)
    
    def _generate_rag_response(self, user_utterance: str, input_iu: dict):
        """RAG機能を使った応答生成"""
        try:
            # RAG検索実行
            if self.rag_enabled and self.rag_retriever:
                start_time = time.time()
                rag_result = self.rag_retriever.retrieve(user_utterance)
                response_time = time.time() - start_time
                
                # 統計更新
                self._update_rag_stats(rag_result, response_time)
                
                # デバッグ情報送信
                self._send_rag_debug_info(user_utterance, rag_result)
            else:
                rag_result = {"level": 3, "type": "general", "content": "", "confidence": 0.3}
            
            # RAG対応LLMで応答生成
            if self.rag_enabled and self.rag_retriever:
                llm = RAGResponseChatGPT(self.config, self.prompts, self.rag_retriever)
            else:
                # フォールバック
                llm = self._create_minimal_llm()
            
            # LLMバッファに追加
            if hasattr(self, 'llm_buffer'):
                self.llm_buffer.put(llm)
            
        except Exception as e:
            print(f"RAG応答生成エラー: {e}")
    
    def _create_minimal_llm(self):
        """最小限のLLMオブジェクトを作成"""
        class MinimalLLM:
            def __init__(self, config, prompts):
                self.config = config
                self.prompts = prompts
                self.user_utterance = ''
                self.asr_time = time.time()
                self.response = iter([
                    {'phrase': 'こんにちは、藤江研究室です。'},
                    {'phrase': 'お手伝いできることがあれば教えてください。'},
                    {'expression': 'normal', 'action': 'wait'}
                ])
            
            def run(self, asr_timestamp, user_utterance, dialogue_history, last_asr_iu_id, parent_llm_buffer):
                self.user_utterance = user_utterance or ''
                self.asr_time = asr_timestamp
                parent_llm_buffer.put(self)
        
        return MinimalLLM(self.config, self.prompts)
    
    def _update_rag_stats(self, rag_result: dict, response_time: float):
        """RAG統計情報を更新"""
        self.rag_stats['queries_processed'] += 1
        
        level = rag_result.get('level', 3)
        if level == 1:
            self.rag_stats['level_1_responses'] += 1
        elif level == 2:
            self.rag_stats['level_2_responses'] += 1
        else:
            self.rag_stats['level_3_responses'] += 1
        
        # 平均応答時間の更新
        total_queries = self.rag_stats['queries_processed']
        current_avg = self.rag_stats['average_response_time']
        self.rag_stats['average_response_time'] = (current_avg * (total_queries - 1) + response_time) / total_queries
    
    def _send_rag_debug_info(self, query: str, rag_result: dict):
        """RAGデバッグ情報を送信"""
        if 'rag_debug' not in self.pub_exchanges:
            return
        
        try:
            debug_info = {
                'query': query[:100],  # 最初の100文字のみ
                'level': rag_result.get('level'),
                'type': rag_result.get('type'),
                'confidence': rag_result.get('confidence'),
                'response_time': rag_result.get('response_time', 0),
                'timestamp': time.time()
            }
            
            # デバッグ情報の送信（実際の実装では適切なメッセージング）
            print(f"RAG Debug: {debug_info}")
            
        except Exception as e:
            print(f"デバッグ情報送信エラー: {e}")
    
    def _rag_stats_monitor(self):
        """RAG統計監視スレッド"""
        while self._is_running:
            try:
                time.sleep(30)  # 30秒間隔で統計出力
                
                if self.rag_stats['queries_processed'] > 0:
                    # コレクション情報を取得
                    collection_info = self.rag_retriever.get_collection_info() if self.rag_retriever else {}
                    
                    print(f"""
=== RAG統計情報 ===
- 処理済みクエリ: {self.rag_stats['queries_processed']}
- Level 1応答: {self.rag_stats['level_1_responses']} (高精度)
- Level 2応答: {self.rag_stats['level_2_responses']} (中精度)
- Level 3応答: {self.rag_stats['level_3_responses']} (低精度)
- 平均応答時間: {self.rag_stats['average_response_time']:.3f}秒
- 基本知識: {collection_info.get('basic_knowledge', 0)}件
- ウェブ文書: {collection_info.get('web_documents', 0)}件
- 合計知識: {collection_info.get('count', 0)}件
===================""")
                
            except Exception as e:
                print(f"統計監視エラー: {e}")
    
    def _fallback_input_loop(self):
        """フォールバック用の入力ループ"""
        while self._is_running:
            try:
                # 簡単な入力処理（実際の実装では適切なIU処理）
                time.sleep(1)
            except Exception as e:
                print(f"入力ループエラー: {e}")
    
    def _fallback_response_loop(self):
        """フォールバック用の応答ループ"""
        while self._is_running:
            try:
                # 簡単な応答処理
                time.sleep(1)
            except Exception as e:
                print(f"応答ループエラー: {e}")
    
    def _fallback_send_response(self):
        """フォールバック用の応答送信"""
        try:
            if hasattr(self, 'llm_buffer') and not self.llm_buffer.empty():
                llm = self.llm_buffer.get()
                print(f"フォールバック応答: {getattr(llm, 'user_utterance', '不明')}")
        except Exception as e:
            print(f"フォールバック応答送信エラー: {e}")
    
    def get_rag_stats(self) -> dict:
        """RAG統計情報を取得"""
        stats = self.rag_stats.copy()
        
        # 知識ベース情報を追加
        if self.rag_retriever:
            collection_info = self.rag_retriever.get_collection_info()
            stats['knowledge_base_status'] = collection_info.get('status', 'unknown')
            stats['knowledge_base_size'] = collection_info.get('count', 0)
        
        return stats
    
    def update_knowledge_base(self) -> bool:
        """知識ベースを手動更新"""
        if not self.rag_enabled or not self.data_collector:
            print("RAG機能またはデータ収集器が無効化されています")
            return False
        
        try:
            print("知識ベース更新を開始...")
            print("藤江研究室のウェブサイトから最新情報を収集しています...")
            
            # ウェブサイトから最新データを収集
            documents = self.data_collector.collect_website_data(max_pages=20)
            
            if documents and self.rag_retriever:
                success = self.rag_retriever.add_documents(documents)
                if success:
                    self.rag_stats['knowledge_base_size'] += len(documents)
                    print(f"知識ベース更新完了: {len(documents)}件の文書を追加")
                    
                    # 更新された内容の概要を表示
                    print("更新された内容:")
                    for i, doc in enumerate(documents[:5]):  # 最初の5件のみ
                        title = doc.get('title', '無題')
                        source = doc.get('source', '')
                        print(f"  {i+1}. {title} (from: {source})")
                    
                    return True
                else:
                    print("文書の追加に失敗しました")
                    return False
            else:
                print("更新する文書が見つかりませんでした")
                return False
                
        except Exception as e:
            print(f"知識ベース更新エラー: {e}")
            return False




def main():
    """RAG対話システムのメイン実行"""
    rag_dialogue = None
    try:
        # 設定読み込み
        config_path = '../config/config.yaml'
        api_config_path = '../config/api_config.yaml'
        
        config = {}
        
        # メイン設定ファイルの読み込み
        if os.path.exists(config_path):
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
        
        # API設定ファイルの読み込み
        if os.path.exists(api_config_path):
            import yaml
            with open(api_config_path, 'r', encoding='utf-8') as f:
                api_config = yaml.safe_load(f) or {}
                # API設定をメイン設定にマージ
                config.update(api_config)
        
        # デフォルト設定の補完
        if 'RAG' not in config:
            config['RAG'] = {'enable': True}
        if 'DIALOGUE' not in config:
            config['DIALOGUE'] = {'history_length': 10}
        if 'ChatGPT' not in config:
            config['ChatGPT'] = {'prompts': {}}
        
        print(f"設定読み込み完了: ChatGPT APIキー = {'設定済み' if config.get('ChatGPT', {}).get('api_key') else '未設定'}")
        
        # RAG対話システム起動
        rag_dialogue = RAGDialogue(config=config)
        rag_dialogue.run()
        
        # メインスレッドを継続実行（Ctrl+Cまで待機）
        print("システムが実行中です。終了するにはCtrl+Cを押してください。")
        while rag_dialogue._is_running:
            time.sleep(1)
        
    except KeyboardInterrupt:
        print("\nシステムを終了します...")
        if rag_dialogue:
            rag_dialogue._is_running = False
    except Exception as e:
        print(f"システム実行エラー: {e}")
        if rag_dialogue:
            rag_dialogue._is_running = False


if __name__ == '__main__':
    main()
