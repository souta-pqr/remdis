import sys
import os
import time
import threading
from typing import Dict, Any, Optional

# RAG関連のインポート
try:
    from rag_retriever import RAGRetriever
    from rag_llm import RAGResponseChatGPT
    from data_collector import FujielabDataCollector
except ImportError as e:
    print(f"Warning: RAGモジュールのインポートに失敗: {e}")
    RAGRetriever = None
    RAGResponseChatGPT = None
    FujielabDataCollector = None

# 既存モジュールからのインポート
try:
    from llm import ResponseChatGPT
except ImportError:
    ResponseChatGPT = None


class RAGHelper:
    """RAG機能をdialogue.pyに統合するためのヘルパークラス"""
    
    def __init__(self, config: Dict[str, Any], prompts: Dict[str, str]):
        self.config = config
        self.prompts = prompts
        
        # RAG機能の有効性チェック
        self.rag_enabled = self._check_rag_availability()
        
        # RAGコンポーネントの初期化
        self.rag_retriever = None
        self.data_collector = None
        
        if self.rag_enabled:
            self._initialize_rag_components()
        
        print(f"RAGHelper初期化完了: RAG機能={'有効' if self.rag_enabled else '無効'}")
    
    def _check_rag_availability(self) -> bool:
        """RAG機能が利用可能かチェック"""
        # 設定でRAGが無効化されている場合
        if not self.config.get('RAG', {}).get('enable', True):
            print("RAG機能は設定で無効化されています")
            return False
        
        # 必要なモジュールが利用できない場合
        if not RAGRetriever or not RAGResponseChatGPT:
            print("RAG関連モジュールが利用できません")
            return False
        
        # OpenAI APIキーが設定されていない場合
        api_key = self.config.get('ChatGPT', {}).get('api_key')
        if not api_key:
            print("OpenAI APIキーが設定されていません")
            return False
        
        return True
    
    def _initialize_rag_components(self):
        """RAGコンポーネントを初期化"""
        try:
            # RAGRetrieverの初期化
            self.rag_retriever = RAGRetriever(self.config)
            print("RAGRetrieverを初期化しました")
            
            # データ収集器の初期化
            if FujielabDataCollector:
                self.data_collector = FujielabDataCollector(self.config)
                print("データ収集器を初期化しました")
                
                # バックグラウンドで初期データ収集を開始
                self._start_background_data_collection()
            
        except Exception as e:
            print(f"RAGコンポーネント初期化エラー: {e}")
            self.rag_enabled = False
    
    def _start_background_data_collection(self):
        """バックグラウンドでデータ収集を開始"""
        def collect_data():
            try:
                print("バックグラウンドデータ収集を開始...")
                
                # 既存の保存済みデータを読み込み
                data_file = os.path.join(os.path.dirname(__file__), "data", "collected_documents.json")
                if os.path.exists(data_file):
                    try:
                        import json
                        with open(data_file, 'r', encoding='utf-8') as f:
                            existing_documents = json.load(f)
                        
                        if existing_documents:
                            print(f"保存済みデータを読み込み: {len(existing_documents)}件")
                            success = self.rag_retriever.add_documents(existing_documents)
                            if success:
                                print("保存済みデータをRAGシステムに追加しました")
                                return
                    except Exception as e:
                        print(f"保存済みデータ読み込みエラー: {e}")
                
                # 新規データ収集
                documents = self.data_collector.collect_website_data(max_pages=20)
                if documents and self.rag_retriever:
                    success = self.rag_retriever.add_documents(documents)
                    if success:
                        print(f"新規データ収集完了: {len(documents)}件の文書を追加")
                    else:
                        print("文書の追加に失敗しました")
                else:
                    print("有効なデータが収集できませんでした")
                    
            except Exception as e:
                print(f"バックグラウンドデータ収集エラー: {e}")
        
        # バックグラウンドスレッドで実行
        thread = threading.Thread(target=collect_data, daemon=True)
        thread.start()
    
    def create_response_llm(self, config: Dict[str, Any], prompts: Dict[str, str]) -> Any:
        """適切な応答生成LLMを作成"""
        if self.rag_enabled and self.rag_retriever:
            # RAG機能付きLLMを作成
            print("RAG機能付きLLMを作成します")
            return RAGResponseChatGPT(config, prompts, self.rag_retriever)
        else:
            # 通常のLLMを作成
            print("通常のLLMを作成します")
            if ResponseChatGPT:
                return ResponseChatGPT(config, prompts)
            else:
                # フォールバック
                return self._create_fallback_llm(config, prompts)
    
    def _create_fallback_llm(self, config: Dict[str, Any], prompts: Dict[str, str]):
        """フォールバック用のLLMを作成"""
        class FallbackLLM:
            def __init__(self, config, prompts):
                self.config = config
                self.prompts = prompts
                self.user_utterance = ''
                self.asr_time = 0.0
                self.response = None
            
            def run(self, asr_timestamp, user_utterance, dialogue_history, last_asr_iu_id, parent_llm_buffer):
                self.user_utterance = user_utterance or ''
                self.asr_time = asr_timestamp
                
                # 簡単なフォールバック応答
                if user_utterance:
                    if any(greeting in user_utterance.lower() for greeting in ['こんにちは', 'hello', 'hi']):
                        phrase = "こんにちは！藤江研究室です。"
                    elif '研究' in user_utterance:
                        phrase = "音声対話技術の研究を行っています。"
                    else:
                        phrase = "ご質問いただき、ありがとうございます。"
                else:
                    phrase = "何かご質問はありますか？"
                
                # 応答を設定
                self.response = iter([{
                    'phrase': phrase,
                    'expression': 'normal',
                    'action': 'wait'
                }])
                
                parent_llm_buffer.put(self)
        
        return FallbackLLM(config, prompts)
    
    def get_rag_stats(self) -> Dict[str, Any]:
        """RAG統計情報を取得"""
        if not self.rag_enabled or not self.rag_retriever:
            return {
                'rag_enabled': False,
                'status': 'disabled'
            }
        
        try:
            collection_info = self.rag_retriever.get_collection_info()
            return {
                'rag_enabled': True,
                'status': 'active',
                'knowledge_base_size': collection_info.get('count', 0),
                'collection_status': collection_info.get('status', 'unknown')
            }
        except Exception as e:
            return {
                'rag_enabled': True,
                'status': 'error',
                'error': str(e)
            }
    
    def update_knowledge_base(self) -> bool:
        """知識ベースを手動更新"""
        if not self.rag_enabled or not self.data_collector:
            print("RAG機能またはデータ収集器が無効です")
            return False
        
        try:
            print("知識ベース更新を開始...")
            documents = self.data_collector.collect_website_data(max_pages=10)
            
            if documents and self.rag_retriever:
                success = self.rag_retriever.add_documents(documents)
                if success:
                    print(f"知識ベース更新完了: {len(documents)}件の文書を追加")
                    return True
            
            print("更新する文書が見つかりませんでした")
            return False
            
        except Exception as e:
            print(f"知識ベース更新エラー: {e}")
            return False


# 設定ファイル読み込み用ヘルパー関数
def load_rag_config(base_config_path: str = '../config/config.yaml', 
                   api_config_path: str = '../config/api_config.yaml') -> Dict[str, Any]:
    """RAG用の設定を読み込み"""
    import yaml
    
    config = {}
    
    # メイン設定ファイル読み込み
    if os.path.exists(base_config_path):
        with open(base_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
    
    # API設定ファイル読み込み
    if os.path.exists(api_config_path):
        with open(api_config_path, 'r', encoding='utf-8') as f:
            api_config = yaml.safe_load(f) or {}
        
        # 設定をマージ
        for key, value in api_config.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value
    
    # デフォルト設定を補完
    if 'RAG' not in config:
        config['RAG'] = {'enable': True}
    
    if 'ChatGPT' not in config:
        config['ChatGPT'] = {
            'prompts': {
                'RESP': 'prompt/rag_system.txt',
                'TO': 'prompt/time_out.txt'
            }
        }
    
    return config


# デバッグ用のテスト関数
def test_rag_helper():
    """RAGHelperのテスト"""
    print("RAGHelperテストを開始...")
    
    # テスト用設定
    test_config = {
        'RAG': {'enable': True},
        'ChatGPT': {
            'api_key': 'test-key',
            'max_tokens': 32,
            'response_generation_model': 'gpt-3.5-turbo',
            'prompts': {
                'RESP': 'テスト用プロンプト',
                'TO': 'タイムアウト用プロンプト'
            }
        }
    }
    
    test_prompts = {
        'RESP': 'テスト用応答プロンプト',
        'TO': 'テスト用タイムアウトプロンプト'
    }
    
    # RAGHelper初期化
    rag_helper = RAGHelper(test_config, test_prompts)
    
    # 統計情報取得
    stats = rag_helper.get_rag_stats()
    print(f"RAG統計: {stats}")
    
    # LLM作成テスト
    llm = rag_helper.create_response_llm(test_config, test_prompts)
    print(f"作成されたLLM: {type(llm)}")
    
    print("RAGHelperテスト完了")


if __name__ == '__main__':
    test_rag_helper()
