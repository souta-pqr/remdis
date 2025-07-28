import sys
import threading
import queue
import time
import re
import os

# 既存モジュールからのインポート
try:
    from dialogue import Dialogue
    from base import RemdisModule, RemdisState, RemdisUtil, RemdisUpdateType
    import prompt.util as prompt_util
except ImportError as e:
    print(f"Warning: 既存モジュールのインポートに失敗: {e}")
    # フォールバック用の基本クラス
    class Dialogue:
        def __init__(self, **kwargs):
            pass

# RAGモジュールからのインポート
try:
    from rag_retriever import RAGRetriever
    from data_collector import FujielabDataCollector
    from rag_llm import RAGResponseChatGPT
except ImportError as e:
    print(f"Warning: RAGモジュールのインポートに失敗: {e}")
    # モック用のダミークラス
    class RAGRetriever:
        def __init__(self, config):
            self.config = config
        def retrieve(self, query):
            return {"level": 3, "type": "general", "content": "モックレスポンス", "confidence": 0.3}
        def add_documents(self, docs):
            return True
    
    class FujielabDataCollector:
        def __init__(self, config):
            self.config = config
        def collect_website_data(self):
            return []
    
    class RAGResponseChatGPT:
        def __init__(self, config, prompts, rag_retriever):
            self.config = config
            self.prompts = prompts
            self.rag_retriever = rag_retriever
        def run(self, *args):
            pass


class RAGDialogue(Dialogue):
    """RAG機能付き対話モジュール"""
    
    def __init__(self, 
                 pub_exchanges=['dialogue', 'dialogue2', 'rag_debug'],
                 sub_exchanges=['asr', 'vap', 'tts', 'bc', 'emo_act'],
                 **kwargs):
        # configをkwargsから取り出し、super().__init__には渡さない
        config = kwargs.pop('config', None)
        self.config = config if config is not None else {
            'DIALOGUE': {'history_length': 10, 'response_generation_interval': 2},
            'RAG': {'enable': True},
            'ChatGPT': {'prompts': {}}
        }

        # 親クラス初期化（存在する場合）
        if hasattr(super(), '__init__'):
            super().__init__(pub_exchanges=pub_exchanges, 
                           sub_exchanges=sub_exchanges, **kwargs)
        else:
            # フォールバック初期化
            self._initialize_fallback(pub_exchanges, sub_exchanges, kwargs)

        # RAG機能の初期化
        self._initialize_rag_components()

        # RAG統計情報
        self.rag_stats = {
            'queries_processed': 0,
            'level_1_responses': 0,
            'level_2_responses': 0,
            'level_3_responses': 0,
            'average_response_time': 0.0,
            'knowledge_base_size': 0
        }

        print("RAG対話モジュールを初期化しました")
    
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
            self.rag_enabled = self.config.get('RAG', {}).get('enable', True)
            
            if not self.rag_enabled:
                print("RAG機能は無効化されています")
                self.rag_retriever = None
                self.data_collector = None
                return
            
            # RAG検索エンジン初期化
            self.rag_retriever = RAGRetriever(self.config)
            
            # データ収集器初期化
            self.data_collector = FujielabDataCollector(self.config)
            
            # 初期データ収集（バックグラウンドで実行）
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
                documents = self.data_collector.collect_website_data(max_pages=10)
                
                if documents and self.rag_retriever:
                    success = self.rag_retriever.add_documents(documents)
                    if success:
                        self.rag_stats['knowledge_base_size'] = len(documents)
                        print(f"初期データ収集完了: {len(documents)}件の文書を追加")
                    else:
                        print("文書の追加に失敗しました")
                else:
                    print("初期データ収集: 有効な文書が見つかりませんでした")
                        
            except Exception as e:
                print(f"初期データ収集エラー: {e}")
        
        # バックグラウンドスレッドで実行
        thread = threading.Thread(target=collect_initial_data, daemon=True)
        thread.start()
    
    def run(self):
        """メインループ（親クラスのメソッドをオーバーライド）"""
        if hasattr(super(), 'run'):
            # 既存のDialogueクラスのrunメソッドがある場合
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
            thread.daemon = True
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
                    print(f"""
RAG統計情報:
- 処理済みクエリ: {self.rag_stats['queries_processed']}
- Level 1応答: {self.rag_stats['level_1_responses']}
- Level 2応答: {self.rag_stats['level_2_responses']}  
- Level 3応答: {self.rag_stats['level_3_responses']}
- 平均応答時間: {self.rag_stats['average_response_time']:.3f}秒
- 知識ベースサイズ: {self.rag_stats['knowledge_base_size']}
                    """)
                
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
            print("RAG機能が無効化されています")
            return False
        
        try:
            print("知識ベース更新を開始...")
            documents = self.data_collector.collect_website_data()
            
            if documents and self.rag_retriever:
                success = self.rag_retriever.add_documents(documents)
                if success:
                    self.rag_stats['knowledge_base_size'] += len(documents)
                    print(f"知識ベース更新完了: {len(documents)}件の文書を追加")
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


# 単体テスト
if __name__ == "__main__":
    import unittest
    from unittest.mock import Mock, patch
    
    class TestRAGDialogue(unittest.TestCase):
        
        def setUp(self):
            """テスト用の設定"""
            self.config = {
                'DIALOGUE': {
                    'history_length': 10,
                    'response_generation_interval': 2
                },
                'RAG': {
                    'enable': True,
                    'vector_db_path': './test_data/chromadb',
                    'top_k': 3
                },
                'ChatGPT': {
                    'api_key': 'test-key',
                    'prompts': {}
                },
                'DATA_SOURCES': {
                    'fujielab_website': 'https://example.com'
                }
            }
        
        def test_initialization(self):
            """初期化テスト"""
            rag_dialogue = RAGDialogue(config=self.config)
            
            self.assertIsInstance(rag_dialogue.rag_stats, dict)
            self.assertTrue(hasattr(rag_dialogue, 'rag_enabled'))
            self.assertTrue(hasattr(rag_dialogue, 'rag_retriever'))
        
        def test_rag_disabled(self):
            """RAG無効化テスト"""
            config = self.config.copy()
            config['RAG']['enable'] = False
            
            rag_dialogue = RAGDialogue(config=config)
            
            self.assertFalse(rag_dialogue.rag_enabled)
            self.assertIsNone(rag_dialogue.rag_retriever)
        
        def test_update_rag_stats(self):
            """RAG統計更新テスト"""
            rag_dialogue = RAGDialogue(config=self.config)
            
            # 初期状態確認
            self.assertEqual(rag_dialogue.rag_stats['queries_processed'], 0)
            
            # 統計更新
            rag_result = {'level': 1, 'type': 'structured', 'confidence': 0.9}
            rag_dialogue._update_rag_stats(rag_result, 0.5)
            
            self.assertEqual(rag_dialogue.rag_stats['queries_processed'], 1)
            self.assertEqual(rag_dialogue.rag_stats['level_1_responses'], 1)
            self.assertEqual(rag_dialogue.rag_stats['average_response_time'], 0.5)
        
        def test_get_rag_stats(self):
            """RAG統計取得テスト"""
            rag_dialogue = RAGDialogue(config=self.config)
            
            stats = rag_dialogue.get_rag_stats()
            
            self.assertIn('queries_processed', stats)
            self.assertIn('level_1_responses', stats)
            self.assertIn('knowledge_base_status', stats)
        
        def test_fallback_initialization(self):
            """フォールバック初期化テスト"""
            # 既存モジュールが利用できない場合のテスト
            rag_dialogue = RAGDialogue(config=self.config)
            
            # 基本的な属性が設定されているかチェック
            self.assertTrue(hasattr(rag_dialogue, 'dialogue_history'))
            self.assertTrue(hasattr(rag_dialogue, 'input_iu_buffer'))
            self.assertTrue(hasattr(rag_dialogue, 'state'))
        
        @patch('threading.Thread')
        def test_run_method(self, mock_thread):
            """runメソッドテスト"""
            rag_dialogue = RAGDialogue(config=self.config)
            
            # エラーが発生しないことを確認
            try:
                rag_dialogue.run()
                # スレッドが作成されることを確認
                self.assertTrue(mock_thread.called)
            except Exception as e:
                self.fail(f"runメソッドでエラーが発生: {e}")
        
        def test_create_minimal_llm(self):
            """最小LLM作成テスト"""
            rag_dialogue = RAGDialogue(config=self.config)
            
            llm = rag_dialogue._create_minimal_llm()
            
            self.assertTrue(hasattr(llm, 'run'))
            self.assertTrue(hasattr(llm, 'user_utterance'))
            self.assertTrue(hasattr(llm, 'response'))
        
        def test_update_knowledge_base(self):
            """知識ベース更新テスト"""
            rag_dialogue = RAGDialogue(config=self.config)
            
            # RAGが有効な場合
            if rag_dialogue.rag_enabled:
                # モック環境では実際の更新は行わないが、エラーが発生しないことを確認
                try:
                    result = rag_dialogue.update_knowledge_base()
                    self.assertIsInstance(result, bool)
                except Exception as e:
                    self.fail(f"知識ベース更新でエラー: {e}")
            else:
                # RAGが無効な場合はFalseを返す
                result = rag_dialogue.update_knowledge_base()
                self.assertFalse(result)
    
    # テスト実行
    print("RAGDialogueの単体テストを実行中...")
    unittest.main(argv=[''], exit=False, verbosity=2)


def main():
    """RAG対話システムのメイン実行"""
    try:
        # 設定読み込み
        config_path = '../config/config.yaml'
        if os.path.exists(config_path):
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            print("設定ファイルが見つかりません。デフォルト設定を使用します。")
            config = {
                'RAG': {'enable': True},
                'DIALOGUE': {'history_length': 10},
                'ChatGPT': {'prompts': {}}
            }
        
        # RAG対話システム起動
        rag_dialogue = RAGDialogue(config=config)
        rag_dialogue.run()
        
    except KeyboardInterrupt:
        print("\nシステムを終了します...")
    except Exception as e:
        print(f"システム実行エラー: {e}")


if __name__ == '__main__':
    main()
