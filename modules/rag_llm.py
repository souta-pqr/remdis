import sys
import json
import time
import re
import string
from typing import Dict, List, Any, Optional, Iterator

# 外部ライブラリ
try:
    import openai
except ImportError:
    print("Warning: openai not installed. Run: pip install openai")
    openai = None

# 既存モジュールからのインポート
try:
    from base import MMDAgentEXLabel
    from llm import ResponseGenerator, ResponseChatGPT
except ImportError as e:
    print(f"Warning: 既存モジュールのインポートに失敗: {e}")
    # フォールバック用のダミークラス
    class MMDAgentEXLabel:
        id2expression = {0: 'normal', 1: 'joy', 2: 'impressed'}
        id2action = {0: 'wait', 1: 'listening', 2: 'nod'}
    
    class ResponseGenerator:
        def __init__(self, config, asr_timestamp, query, dialogue_history, prompts):
            pass
        def __next__(self):
            raise StopIteration
        def __iter__(self):
            return self
    
    class ResponseChatGPT:
        def __init__(self, config, prompts):
            self.config = config
            self.prompts = prompts

# RAGモジュールからのインポート
try:
    from rag_retriever import RAGRetriever
except ImportError:
    print("Warning: RAGRetrieverのインポートに失敗")
    class RAGRetriever:
        def retrieve(self, query):
            return {"level": 3, "type": "general", "content": "", "confidence": 0.3}


class RAGResponseGenerator(ResponseGenerator):
    """RAG機能付き応答生成器"""
    
    def __init__(self, config: Dict[str, Any], asr_timestamp: float, 
                 query: Optional[str], dialogue_history: List[Dict], 
                 prompts: Dict[str, str], rag_retriever: RAGRetriever):
        
        # 基本設定
        self.max_tokens = config['ChatGPT']['max_tokens']
        self.max_message_num_in_context = config['ChatGPT']['max_message_num_in_context']
        self.model = config['ChatGPT']['response_generation_model']
        
        # 処理対象のユーザ発話に関する情報
        self.asr_timestamp = asr_timestamp
        self.query = query
        self.dialogue_history = dialogue_history
        self.prompts = prompts
        self.rag_retriever = rag_retriever
        
        # 生成中の応答を保持・パースする変数
        self.response_fragment = ''
        self.punctuation_pattern = re.compile('[、。！？]')
        
        # RAG検索結果
        self.rag_result = None
        self.rag_context = ""
        
        # ChatGPTの応答ストリーム
        self.response = None
        
        # RAG機能付きメッセージ構築と応答生成
        self._initialize_rag_response()
    
    def _initialize_rag_response(self):
        """RAG機能付き応答を初期化"""
        try:
            # RAG検索実行
            if self.query and self.rag_retriever:
                self.rag_result = self.rag_retriever.retrieve(self.query)
                self.log(f"RAG検索完了: Level {self.rag_result.get('level')}, Type {self.rag_result.get('type')}")
            else:
                self.rag_result = {"level": 3, "type": "general", "content": "", "confidence": 0.3}
            
            # RAGコンテキスト構築
            self.rag_context = self._build_rag_context(self.rag_result)
            
            # ChatGPTに入力するメッセージ構築
            messages = self._build_messages_with_rag()
            
            self.log(f"Call ChatGPT with RAG: {self.query=}")
            
            # ChatGPTに対話文脈を入力してストリーミング形式で応答の生成を開始
            if openai:
                self.response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    stream=True
                )
            else:
                # モック応答（テスト用）
                self.response = self._create_mock_response()
            
        except Exception as e:
            self.log(f"RAG応答初期化エラー: {e}")
            # エラー時はフォールバック
            self.response = self._create_fallback_response()
    
    def _build_rag_context(self, rag_result: Dict[str, Any]) -> str:
        """RAG検索結果からコンテキストを構築"""
        if not rag_result:
            return ""
        
        level = rag_result.get('level', 3)
        content = rag_result.get('content', '')
        confidence = rag_result.get('confidence', 0.0)
        
        if level == 1:
            # Level 1: 構造化基本情報
            return f"""
【確実な基本情報】
{content}
※この情報は確認済みです。自信を持って回答してください。
信頼度: {confidence*100:.0f}%
"""
        elif level == 2:
            # Level 2: RAG検索結果
            if isinstance(content, list) and len(content) > 0:
                # 複数の検索結果を統合
                documents = content[:3]  # 上位3件のみ使用
                context_text = '\n'.join([f"- {doc[:200]}..." if len(doc) > 200 else f"- {doc}" for doc in documents])
                
                return f"""
【研究室関連情報】
{context_text}
※この情報は研究室の最新データから取得されました。
信頼度: {confidence*100:.0f}%
"""
            else:
                return f"""
【研究室関連情報】
{content}
信頼度: {confidence*100:.0f}%
"""
        else:
            # Level 3: 一般知識フォールバック
            return """
【注意】
研究室固有の最新情報が見つかりませんでした。
一般的な知識で回答しますが、詳細は研究室ウェブサイト（https://www.fujielab.org/）をご確認ください。
"""
    
    def _build_messages_with_rag(self) -> List[Dict[str, str]]:
        """RAGコンテキスト付きメッセージを構築"""
        messages = []
        
        # システムプロンプト（RAGコンテキスト付き）
        system_prompt = self._build_system_prompt()
        messages.append({"role": "system", "content": system_prompt})
        
        # 過去の対話履歴を対話文脈に追加
        i = max(0, len(self.dialogue_history) - self.max_message_num_in_context)
        messages.extend(self.dialogue_history[i:])
        
        # プロンプトおよび新しいユーザ発話を対話文脈に追加
        if self.query:
            messages.extend([
                {'role': 'user', 'content': self.prompts.get('RESP', '応答してください')},
                {'role': 'system', 'content': "OK"},
                {'role': 'user', 'content': self.query}
            ])
        else:
            # 新しいユーザ発話が存在せず自ら発話する場合
            messages.extend([
                {'role': 'user', 'content': self.prompts.get('TO', '話しかけてください')}
            ])
        
        return messages
    
    def _build_system_prompt(self) -> str:
        """RAGコンテキスト付きシステムプロンプトを構築"""
        base_prompt = """あなたは藤江研究室（fujielab）の案内AIアシスタントです。
名古屋大学大学院情報学研究科に所属し、藤江信也教授が主宰する研究室です。

研究分野：
- 音声言語処理
- 対話システム
- 自然言語処理
- マルチモーダル対話

以下の情報を参考に、正確で親しみやすい回答をしてください：
"""
        
        # RAGコンテキストを追加
        if self.rag_context:
            system_prompt = base_prompt + "\n" + self.rag_context
        else:
            system_prompt = base_prompt
        
        # 応答フォーマット指示を追加
        system_prompt += """

回答は句読点（、。！？）で分割して出力してください。
最後にアシスタントの感情（0_平静,1_喜び,2_感動,3_納得,4_考え中,5_眠い,6_ジト目,7_同情,8_恥ずかしい,9_怒り）と
動き（0_待機,1_ユーザの声に気づく,2_うなずく,3_首をかしげる,4_考え中,5_会釈,6_お辞儀,7_片手を振る,8_両手を振る,9_見渡す）を
以下のフォーマットで出力してください：

応答文/感情_動き

例：こんにちは、藤江研究室です。/1_喜び,2_うなずく
"""
        
        return system_prompt
    
    def _create_mock_response(self) -> Iterator[Dict[str, Any]]:
        """モック応答を作成（テスト用）"""
        mock_phrases = [
            "こんにちは、",
            "藤江研究室です。",
            "音声言語処理を研究しています。",
            "/1_喜び,2_うなずく"
        ]
        
        for phrase in mock_phrases:
            yield {
                'choices': [{
                    'delta': {
                        'content': phrase
                    }
                }]
            }
        
        # 終了マーカー
        yield {
            'choices': [{
                'delta': {}
            }]
        }
    
    def _create_fallback_response(self) -> Iterator[Dict[str, Any]]:
        """フォールバック応答を作成"""
        fallback_text = "申し訳ございませんが、システムエラーが発生しました。/4_考え中,3_首をかしげる"
        
        yield {
            'choices': [{
                'delta': {
                    'content': fallback_text
                }
            }]
        }
        
        yield {
            'choices': [{
                'delta': {}
            }]
        }
    
    def __next__(self) -> Dict[str, Any]:
        """応答の断片を順次返す（既存のResponseGeneratorと同じインターフェース）"""
        # 引数（例: '1_喜び,6_会釈'）をパースして，expressionとactionを取得
        def _parse_split(split):
            expression = MMDAgentEXLabel.id2expression[0]
            action = MMDAgentEXLabel.id2action[0]

            # expression/actionを取得
            if "," in split:
                expression, action = split.split(",", 1)

                expression = expression.split("_")[0]
                expression = int(expression) if expression.isdigit() else 0
                expression = MMDAgentEXLabel.id2expression.get(expression, MMDAgentEXLabel.id2expression[0])

                action = action.split("_")[0]
                action = int(action) if action.isdigit() else 0
                action = MMDAgentEXLabel.id2action.get(action, MMDAgentEXLabel.id2action[0])

            return {
                "expression": expression,
                "action": action
            }

        # ChatGPTの応答を順次パースして返す
        try:
            for chunk in self.response:
                chunk_message = chunk['choices'][0]['delta']

                if 'content' in chunk_message.keys():
                    new_token = chunk_message.get('content')

                    # 応答の断片を追加
                    if new_token != "/":
                        self.response_fragment += new_token

                    # 句読点で応答を分割
                    splits = self.punctuation_pattern.split(self.response_fragment, 1)

                    # 次のループのために残りの断片を保持
                    self.response_fragment = splits[-1]

                    # 句読点が存在していた場合は1つ目の断片を返す
                    if len(splits) == 2 or new_token == "/":
                        if splits[0]:
                            return {"phrase": splits[0]}
                    
                    # 応答の最後が来た場合は残りの断片を返す
                    if new_token == "/":
                        if self.response_fragment:
                            return {"phrase": self.response_fragment}
                        self.response_fragment = ''
                else:
                    # ChatGPTの応答が完了した場合は残りの断片をパースして返す
                    if self.response_fragment:
                        return _parse_split(self.response_fragment)

        except Exception as e:
            self.log(f"応答生成エラー: {e}")
            return {"phrase": "エラーが発生しました"}

        raise StopIteration
    
    def __iter__(self):
        return self
    
    def log(self, *args, **kwargs):
        """デバッグ用のログ出力"""
        print(f"[RAG-{time.time():.5f}]", *args, flush=True, **kwargs)


class RAGResponseChatGPT(ResponseChatGPT):
    """RAG機能付きChatGPT応答生成クラス"""
    
    def __init__(self, config: Dict[str, Any], prompts: Dict[str, str], 
                 rag_retriever: RAGRetriever):
        
        # 親クラス初期化（存在する場合）
        if hasattr(super(), '__init__'):
            super().__init__(config, prompts)
        else:
            # フォールバック初期化
            self.config = config
            self.prompts = prompts
            
            # OpenAI API設定
            if openai:
                openai.api_key = config.get('ChatGPT', {}).get('api_key', '')
            
            # 入力されたユーザ発話に関する情報を保持する変数
            self.user_utterance = ''
            self.response = ''
            self.last_asr_iu_id = ''
            self.asr_time = 0.0
        
        # RAG機能追加
        self.rag_retriever = rag_retriever
        
        # RAG統計
        self.rag_usage_stats = {
            'total_requests': 0,
            'level_1_usage': 0,
            'level_2_usage': 0,
            'level_3_usage': 0,
            'average_confidence': 0.0
        }
    
    def run(self, asr_timestamp: float, user_utterance: Optional[str], 
            dialogue_history: List[Dict], last_asr_iu_id: Optional[str], 
            parent_llm_buffer):
        """RAG検索付き応答生成を開始"""
        
        self.user_utterance = user_utterance or ""
        self.last_asr_iu_id = last_asr_iu_id
        self.asr_time = asr_timestamp
        
        # RAG統計更新
        self._update_usage_stats()
        
        try:
            # RAG対応応答生成器を作成
            self.response = RAGResponseGenerator(
                self.config, 
                asr_timestamp, 
                user_utterance, 
                dialogue_history, 
                self.prompts,
                self.rag_retriever
            )
            
            # 自身をDialogueモジュールが持つLLMバッファに追加
            parent_llm_buffer.put(self)
            
        except Exception as e:
            print(f"RAG応答生成開始エラー: {e}")
            # フォールバック処理
            self._create_fallback_response(asr_timestamp, user_utterance, dialogue_history)
            parent_llm_buffer.put(self)
    
    def _update_usage_stats(self):
        """RAG使用統計を更新"""
        self.rag_usage_stats['total_requests'] += 1
    
    def _create_fallback_response(self, asr_timestamp: float, user_utterance: Optional[str], 
                                dialogue_history: List[Dict]):
        """フォールバック応答を作成"""
        try:
            # 既存のResponseGeneratorを使用（RAG無し）
            self.response = ResponseGenerator(
                self.config, 
                asr_timestamp, 
                user_utterance, 
                dialogue_history, 
                self.prompts
            )
        except:
            # 最終フォールバック
            self.response = self._create_minimal_response(user_utterance)
    
    def _create_minimal_response(self, user_utterance: Optional[str]):
        """最小限の応答を作成"""
        class MinimalResponse:
            def __init__(self, utterance):
                self.utterance = utterance or "無言"
                self.phrases = [
                    f"お疲れさまです。",
                    f"「{self.utterance[:50]}」について考えています。",
                    f"詳細は研究室ウェブサイトをご確認ください。"
                ]
                self.index = 0
            
            def __next__(self):
                if self.index < len(self.phrases):
                    phrase = self.phrases[self.index]
                    self.index += 1
                    return {"phrase": phrase}
                elif self.index == len(self.phrases):
                    self.index += 1
                    return {"expression": "normal", "action": "wait"}
                else:
                    raise StopIteration
            
            def __iter__(self):
                return self
        
        return MinimalResponse(user_utterance)
    
    def get_rag_stats(self) -> Dict[str, Any]:
        """RAG使用統計を取得"""
        return self.rag_usage_stats.copy()


# 単体テスト
if __name__ == "__main__":
    import unittest
    from unittest.mock import Mock, patch
    
    class TestRAGResponseGenerator(unittest.TestCase):
        
        def setUp(self):
            """テスト用の設定"""
            self.config = {
                'ChatGPT': {
                    'max_tokens': 64,
                    'max_message_num_in_context': 3,
                    'response_generation_model': 'gpt-3.5-turbo',
                    'api_key': 'test-key'
                }
            }
            
            self.prompts = {
                'RESP': 'あなたは対話AIです。応答してください。',
                'TO': '話しかけてください。'
            }
            
            # モックRAGRetriever
            self.mock_rag_retriever = Mock()
            self.mock_rag_retriever.retrieve.return_value = {
                'level': 1,
                'type': 'structured',
                'content': '藤江研究室（fujielab）は音声言語処理、対話システム、自然言語処理の研究を行う研究室です。',
                'confidence': 0.95
            }
        
        def test_initialization(self):
            """初期化テスト"""
            generator = RAGResponseGenerator(
                self.config, 
                time.time(), 
                "研究室について教えて", 
                [], 
                self.prompts, 
                self.mock_rag_retriever
            )
            
            self.assertIsNotNone(generator.rag_result)
            self.assertIsNotNone(generator.rag_context)
            self.mock_rag_retriever.retrieve.assert_called_once()
        
        def test_build_rag_context_level1(self):
            """Level 1のRAGコンテキスト構築テスト"""
            generator = RAGResponseGenerator(
                self.config, 
                time.time(), 
                "研究室について", 
                [], 
                self.prompts, 
                self.mock_rag_retriever
            )
            
            self.assertIn('確実な基本情報', generator.rag_context)
            self.assertIn('藤江研究室', generator.rag_context)
            self.assertIn('95%', generator.rag_context)
        
        def test_build_rag_context_level2(self):
            """Level 2のRAGコンテキスト構築テスト"""
            self.mock_rag_retriever.retrieve.return_value = {
                'level': 2,
                'type': 'rag',
                'content': ['文書1の内容', '文書2の内容'],
                'confidence': 0.8
            }
            
            generator = RAGResponseGenerator(
                self.config, 
                time.time(), 
                "詳細な研究内容", 
                [], 
                self.prompts, 
                self.mock_rag_retriever
            )
            
            self.assertIn('研究室関連情報', generator.rag_context)
            self.assertIn('80%', generator.rag_context)
        
        def test_build_rag_context_level3(self):
            """Level 3のRAGコンテキスト構築テスト"""
            self.mock_rag_retriever.retrieve.return_value = {
                'level': 3,
                'type': 'general',
                'content': '一般的な知識で回答',
                'confidence': 0.3
            }
            
            generator = RAGResponseGenerator(
                self.config, 
                time.time(), 
                "不明な質問", 
                [], 
                self.prompts, 
                self.mock_rag_retriever
            )
            
            self.assertIn('注意', generator.rag_context)
            self.assertIn('ウェブサイト', generator.rag_context)
        
        def test_build_messages_with_rag(self):
            """RAGメッセージ構築テスト"""
            dialogue_history = [
                {'role': 'user', 'content': '前の質問'},
                {'role': 'assistant', 'content': '前の回答'}
            ]
            
            generator = RAGResponseGenerator(
                self.config, 
                time.time(), 
                "新しい質問", 
                dialogue_history, 
                self.prompts, 
                self.mock_rag_retriever
            )
            
            messages = generator._build_messages_with_rag()
            
            # メッセージ構造確認
            self.assertGreater(len(messages), 0)
            self.assertEqual(messages[0]['role'], 'system')
            self.assertIn('藤江研究室', messages[0]['content'])
            
            # 対話履歴が含まれているか確認
            user_messages = [msg for msg in messages if msg['role'] == 'user']
            self.assertGreater(len(user_messages), 0)
        
        def test_mock_response_iteration(self):
            """モック応答のイテレーションテスト"""
            generator = RAGResponseGenerator(
                self.config, 
                time.time(), 
                "テスト質問", 
                [], 
                self.prompts, 
                self.mock_rag_retriever
            )
            
            # イテレーションテスト
            responses = []
            try:
                for response in generator:
                    responses.append(response)
                    if len(responses) > 10:  # 無限ループ防止
                        break
            except StopIteration:
                pass
            
            self.assertGreater(len(responses), 0)
            
            # 応答内容確認
            phrase_responses = [r for r in responses if 'phrase' in r]
            self.assertGreater(len(phrase_responses), 0)
    
    class TestRAGResponseChatGPT(unittest.TestCase):
        
        def setUp(self):
            """テスト用の設定"""
            self.config = {
                'ChatGPT': {
                    'max_tokens': 64,
                    'api_key': 'test-key',
                    'response_generation_model': 'gpt-3.5-turbo'
                }
            }
            
            self.prompts = {
                'RESP': '応答してください',
                'TO': '話しかけてください'
            }
            
            self.mock_rag_retriever = Mock()
            self.mock_rag_retriever.retrieve.return_value = {
                'level': 1,
                'type': 'structured',
                'content': 'テスト応答',
                'confidence': 0.9
            }
        
        def test_initialization(self):
            """初期化テスト"""
            rag_chatgpt = RAGResponseChatGPT(
                self.config, 
                self.prompts, 
                self.mock_rag_retriever
            )
            
            self.assertEqual(rag_chatgpt.rag_retriever, self.mock_rag_retriever)
            self.assertIn('total_requests', rag_chatgpt.rag_usage_stats)
        
        def test_run_method(self):
            """runメソッドテスト"""
            rag_chatgpt = RAGResponseChatGPT(
                self.config, 
                self.prompts, 
                self.mock_rag_retriever
            )
            
            mock_buffer = Mock()
            
            # エラーが発生しないことを確認
            try:
                rag_chatgpt.run(
                    time.time(), 
                    "テスト発話", 
                    [], 
                    "test_id", 
                    mock_buffer
                )
                
                # バッファに追加されることを確認
                mock_buffer.put.assert_called_once()
                
            except Exception as e:
                self.fail(f"runメソッドでエラーが発生: {e}")
        
        def test_get_rag_stats(self):
            """RAG統計取得テスト"""
            rag_chatgpt = RAGResponseChatGPT(
                self.config, 
                self.prompts, 
                self.mock_rag_retriever
            )
            
            stats = rag_chatgpt.get_rag_stats()
            
            self.assertIn('total_requests', stats)
            self.assertIn('level_1_usage', stats)
            self.assertEqual(stats['total_requests'], 0)  # 初期値
    
    # テスト実行
    print("RAG LLMモジュールの単体テストを実行中...")
    unittest.main(argv=[''], exit=False, verbosity=2)


def main():
    """RAG LLMの単体実行テスト"""
    config = {
        'ChatGPT': {
            'api_key': 'test-key',
            'max_tokens': 64,
            'max_message_num_in_context': 3,
            'response_generation_model': 'gpt-3.5-turbo'
        }
    }
    
    prompts = {
        'RESP': 'あなたは藤江研究室の案内AIです。応答してください。',
        'TO': '話しかけてください。'
    }
    
    # モックRAGRetriever
    class MockRAGRetriever:
        def retrieve(self, query):
            return {
                'level': 1,
                'type': 'structured',
                'content': '藤江研究室（fujielab）は千葉工業大学の研究室です。',
                'confidence': 0.95
            }
    
    rag_retriever = MockRAGRetriever()
    
    # テスト実行
    print("RAG応答生成テストを開始...")
    
    response_generator = RAGResponseGenerator(
        config, 
        time.time(), 
        "研究室について教えてください", 
        [], 
        prompts, 
        rag_retriever
    )
    
    print("生成された応答:")
    for i, response in enumerate(response_generator):
        print(f"{i+1}: {response}")
        if i > 10:  # 無限ループ防止
            break
    
    print("テスト完了")


if __name__ == '__main__':
    main()
    