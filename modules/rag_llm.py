import sys
import json
import time
import re
import string
import yaml
from typing import Dict, List, Any, Optional, Iterator

# 外部ライブラリ
try:
    import openai
    # OpenAI APIキーの設定
    try:
        import yaml
        config_path = '../config/api_config.yaml'
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                api_config = yaml.safe_load(f) or {}
            if api_config and 'ChatGPT' in api_config and 'api_key' in api_config['ChatGPT']:
                openai.api_key = api_config['ChatGPT']['api_key']
                print(f"OpenAI APIキーを設定しました: {openai.api_key[:10]}...")
            else:
                print("OpenAI APIキーが見つかりません")
        except FileNotFoundError:
            print(f"設定ファイルが見つかりません: {config_path}")
        except Exception as e:
            print(f"設定ファイル読み込みエラー: {e}")
    except Exception as e:
        print(f"OpenAI APIキー設定エラー: {e}")
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
        
        # 設定を最初に保存
        self.config = config
        
        # 基本設定
        self.max_tokens = config['ChatGPT']['max_tokens']
        self.max_message_num_in_context = config['ChatGPT']['max_message_num_in_context']
        self.model = config['ChatGPT']['response_generation_model']
        
        # 設定確認のデバッグ出力
        print(f"[RAG Generator] 設定確認:")
        print(f"  - model: {self.model}")
        print(f"  - max_tokens: {self.max_tokens}")
        print(f"  - max_message_num_in_context: {self.max_message_num_in_context}")
        
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
            # エラーフラグを初期化
            self._error_occurred = False
            self._response_generated = False  # 応答生成フラグを初期化
            self._response_complete = False   # 応答完了フラグを初期化
            
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
            
            self.log(f"Call ChatGPT with RAG: {self.query=}, model={self.model}")
            self.log(f"送信メッセージ数: {len(messages)}")
            for i, msg in enumerate(messages):
                self.log(f"メッセージ{i+1}: {msg['role']} = {msg['content'][:100]}...")
            
            # ChatGPTが利用可能でAPIキーが設定されている場合
            api_key = None
            if openai:
                # APIキーを取得
                api_key = self.config.get('ChatGPT', {}).get('api_key')
                if not api_key:
                    try:
                        # 環境変数から取得を試行
                        import os
                        api_key = os.environ.get('OPENAI_API_KEY')
                    except:
                        pass
            
            if openai and api_key:
                try:
                    # OpenAI APIライブラリのバージョンを確認して適切な方法を選択
                    try:
                        # 新しいOpenAI API (v1.0+) でのクライアント作成を試行
                        if hasattr(openai, 'OpenAI'):
                            client = openai.OpenAI(api_key=api_key)
                            
                            # ChatGPTに対話文脈を入力してストリーミング形式で応答の生成を開始
                            self.response = client.chat.completions.create(
                                model=self.model,
                                messages=messages,
                                max_tokens=self.max_tokens,
                                stream=True,
                                temperature=0.7
                            )
                            self.log(f"ChatGPT応答ストリーム開始 (新API v1.0+)")
                        else:
                            # 古いAPIを使用
                            openai.api_key = api_key
                            self.response = openai.ChatCompletion.create(
                                model=self.model,
                                messages=messages,
                                max_tokens=self.max_tokens,
                                stream=True,
                                temperature=0.7
                            )
                            self.log(f"ChatGPT応答ストリーム開始 (旧API v0.x)")
                    except Exception as api_error:
                        self.log(f"新API失敗、旧API試行: {api_error}")
                        # 新APIが失敗した場合、旧APIを試行
                        openai.api_key = api_key
                        self.response = openai.ChatCompletion.create(
                            model=self.model,
                            messages=messages,
                            max_tokens=self.max_tokens,
                            stream=True,
                            temperature=0.7
                        )
                        self.log(f"ChatGPT応答ストリーム開始 (旧API フォールバック)")
                        
                except Exception as e:
                    self.log(f"ChatGPT API呼び出しエラー: {e}")
                    self.log(f"エラータイプ: {type(e)}")
                    import traceback
                    traceback.print_exc()
                    self.response = self._create_fallback_response()
            else:
                self.log("ChatGPT利用不可、フォールバックモードで動作")
                self.response = self._create_fallback_response()
                
        except Exception as e:
            self.log(f"RAG応答初期化エラー: {e}")
            self._error_occurred = True
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
信頼度: {confidence:.2f}
            """.strip()
        elif level == 2:
            # Level 2: RAG検索結果
            return f"""
【検索結果】
{content}
信頼度: {confidence:.2f}
            """.strip()
        else:
            # Level 3: フォールバック
            return """
【一般的な情報】
藤江研究室は音声言語処理と対話システムの研究を行っている研究室です。
            """.strip()
    
    def _build_messages_with_rag(self) -> List[Dict[str, str]]:
        """RAGコンテキストを含むメッセージを構築"""
        messages = []
        
        # システムプロンプトを構築
        system_prompt = self._build_system_prompt()
        messages.append({"role": "system", "content": system_prompt})
        
        # 対話履歴を追加
        for hist in self.dialogue_history[-self.max_message_num_in_context:]:
            if hist.get('role') == 'user':
                messages.append({"role": "user", "content": hist.get('content', '')})
            elif hist.get('role') == 'assistant':
                messages.append({"role": "assistant", "content": hist.get('content', '')})
        
        # 現在のユーザー発話を追加
        if self.query:
            messages.append({"role": "user", "content": self.query})
        
        return messages
    
    def _build_system_prompt(self) -> str:
        """RAGコンテキストを含むシステムプロンプトを構築"""
        base_prompt = self.prompts.get('RESP', '藤江研究室の案内AIアシスタントとして回答してください。')
        
        # RAGコンテキストをプロンプトに組み込む
        if self.rag_context and '{rag_context}' in base_prompt:
            # プロンプトテンプレートのRAGコンテキストを置換
            return base_prompt.replace('{rag_context}', self.rag_context)
        elif self.rag_context:
            # 従来の方式でRAGコンテキストを追加
            return f"""{base_prompt}

以下の情報を参考にして回答してください：

{self.rag_context}

回答は自然で親しみやすい口調で、関連する情報を適切に活用してください。
出典がある場合は最後に記載してください。
応答の最後に感情と動作を「/感情_動作」の形式で追加してください。
感情: 0_平静, 1_喜び, 2_感動, 3_納得, 4_考え中, 5_眠い, 6_ジト目, 7_同情, 8_恥ずかしい, 9_怒り
動作: 0_待機, 1_ユーザの声に気づく, 2_うなずく, 3_首をかしげる, 4_考え中, 5_会釈, 6_お辞儀, 7_片手を振る, 8_両手を振る, 9_見渡す
"""
        else:
            return base_prompt
    
    def _generate_fallback_text(self) -> str:
        """フォールバックテキストを生成"""
        # RAG検索結果を利用したフォールバック応答
        if hasattr(self, 'rag_result') and self.rag_result and self.rag_result.get('content'):
            rag_content = self.rag_result.get('content', '')
            level = self.rag_result.get('level', 3)
            
            # 挨拶の場合は専用の応答
            if hasattr(self, 'query') and self.query and any(greeting in self.query.lower() for greeting in ['こんにちは', 'hello', 'hi']):
                return "こんにちは！藤江研究室です。音声対話技術について何かご質問がございましたらお聞かせください。/1_喜び,2_うなずく"
            
            # 質問内容に応じた応答を生成
            if hasattr(self, 'query') and self.query:
                query_lower = self.query.lower()
                if 'プロジェクト' in self.query or 'project' in query_lower:
                    return "藤江研究室では会話ロボットを中心とした音声対話システムの研究を行っています。具体的には音声認識、自然言語処理、対話制御などの技術開発に取り組んでいます。/1_喜び,2_うなずく"
                elif '研究' in self.query or 'research' in query_lower:
                    return "藤江研究室は千葉工業大学で音声言語処理と対話システムの研究を行っています。会話ロボットの基礎技術から応用まで幅広く研究を進めています。/1_喜び,2_うなずく"
                elif 'メンバー' in self.query or 'member' in query_lower:
                    return "藤江研究室には藤江真也教授をはじめ、多くの学生が所属し、音声対話技術の研究に取り組んでいます。/1_喜び,2_うなずく"
                else:
                    # 一般的な質問の場合、RAG内容を要約して返答
                    if len(rag_content) > 150:
                        rag_content = rag_content[:150] + "..."
                    return f"藤江研究室についてお答えします。{rag_content}/1_喜び,2_うなずく"
            else:
                return "藤江研究室についてご質問いただき、ありがとうございます。音声対話技術についてお聞かせください。/1_喜び,2_うなずく"
        else:
            # RAG結果がない場合の基本応答
            if hasattr(self, 'query') and self.query and any(greeting in self.query.lower() for greeting in ['こんにちは', 'hello', 'hi']):
                return "こんにちは！藤江研究室です。音声対話技術の研究を行っています。/1_喜び,2_うなずく"
            elif hasattr(self, 'query') and self.query and '研究室' in self.query:
                return "藤江研究室は千葉工業大学で音声言語処理と対話システムの研究を行っています。/1_喜び,2_うなずく"
            else:
                return "藤江研究室についてご質問いただき、ありがとうございます。/1_喜び,2_うなずく"

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
        # RAG検索結果を利用したフォールバック応答
        if hasattr(self, 'rag_result') and self.rag_result and self.rag_result.get('content'):
            rag_content = self.rag_result.get('content', '')
            if len(rag_content) > 200:
                rag_content = rag_content[:200] + "..."
            
            # 挨拶の場合は専用の応答
            if hasattr(self, 'query') and self.query and any(greeting in self.query.lower() for greeting in ['こんにちは', 'hello', 'hi']):
                fallback_text = "こんにちは！藤江研究室です。音声対話技術について何かご質問がございましたらお聞かせください。/1_喜び,2_うなずく"
            else:
                fallback_text = f"藤江研究室についてお答えします。{rag_content}/1_喜び,2_うなずく"
        else:
            # RAG結果がない場合の基本応答
            if hasattr(self, 'query') and self.query and any(greeting in self.query.lower() for greeting in ['こんにちは', 'hello', 'hi']):
                fallback_text = "こんにちは！藤江研究室です。音声対話技術の研究を行っています。/1_喜び,2_うなずく"
            else:
                fallback_text = "申し訳ございませんが、システムエラーが発生しました。/4_考え中,3_首をかしげる"
        
        # フォールバック応答を一度だけ返す
        yield {
            'choices': [{
                'delta': {
                    'content': fallback_text
                }
            }]
        }
        
        # 終了マーカー
        yield {
            'choices': [{
                'delta': {}
            }]
        }
    
    def __next__(self) -> Dict[str, Any]:
        """応答の断片を順次返す（既存のResponseGeneratorと同じインターフェース）"""
        # エラー状態チェック（無限ループ防止）
        if hasattr(self, '_error_occurred') and self._error_occurred:
            raise StopIteration
        
        # 応答が既に完了している場合
        if hasattr(self, '_response_complete') and self._response_complete:
            raise StopIteration
        
        # 引数（例: '1_喜び,6_会釈'）をパースして，expressionとactionを取得
        def _parse_split(split):
            expression = MMDAgentEXLabel.id2expression[0]
            action = MMDAgentEXLabel.id2action[0]
            
            if ',' in split:
                parts = split.split(',')
                for part in parts:
                    part = part.strip()
                    if '_' in part:
                        id_str, name = part.split('_', 1)
                        try:
                            part_id = int(id_str)
                            if name in ['喜び', '感動', '納得', '考え中', '眠い', 'ジト目', '同情', '恥ずかしい', '怒り']:
                                expression = MMDAgentEXLabel.id2expression.get(part_id, expression)
                            elif name in ['待機', 'ユーザの声に気づく', 'うなずく', '首をかしげる', '考え中', '会釈', 'お辞儀', '片手を振る', '両手を振る', '見渡す']:
                                action = MMDAgentEXLabel.id2action.get(part_id, action)
                        except ValueError:
                            pass
            
            return expression, action

        try:
            # ChatGPTからの応答ストリームを処理
            if self.response:
                self.log(f"応答ストリーム処理開始")
                
                # 完全な応答を構築
                full_response = ""
                
                # ストリーミング応答を順次処理して完全な応答を構築
                chunk_count = 0
                for chunk in self.response:
                    chunk_count += 1
                    self.log(f"チャンク{chunk_count}を処理中")
                    
                    # 新しいAPI形式での処理
                    content = ""
                    finish_reason = None
                    try:
                        if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            content = getattr(delta, 'content', '') or ''
                            finish_reason = getattr(chunk.choices[0], 'finish_reason', None)
                            self.log(f"新API: content='{content}', finish_reason='{finish_reason}'")
                        else:
                            # 古いAPI形式での処理
                            if isinstance(chunk, dict) and chunk.get('choices') and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                content = delta.get('content', '') or ''
                                finish_reason = chunk['choices'][0].get('finish_reason')
                                self.log(f"旧API: content='{content}', finish_reason='{finish_reason}'")
                            else:
                                self.log(f"不明なチャンク形式: {type(chunk)}, chunk={chunk}")
                                continue
                    except Exception as e:
                        self.log(f"チャンク処理エラー: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                    
                    if content:
                        full_response += content
                        self._response_generated = True
                        self.log(f"応答構築中: '{full_response}'")
                    
                    # ストリーム終了判定
                    if finish_reason:
                        self.log(f"ストリーム終了: finish_reason={finish_reason}")
                        break
                
                self.log(f"ストリーム処理完了: {chunk_count}チャンク処理、完全応答='{full_response}'")
                
                # 完全な応答が構築された場合、一度だけ返す
                if full_response.strip():
                    self._response_complete = True  # 応答完了フラグを設定
                    
                    # 感情・動作パースを確認
                    if '/' in full_response:
                        phrase, emotion_action = full_response.rsplit('/', 1)
                        expression, action = _parse_split(emotion_action)
                        return {
                            'phrase': phrase.strip(),
                            'expression': expression,
                            'action': action
                        }
                    else:
                        return {'phrase': full_response.strip()}
            else:
                self.log("応答ストリームが存在しません")
            
            # 応答が全く生成されなかった場合のフォールバック
            if not hasattr(self, '_response_generated') or not self._response_generated:
                self.log("応答が生成されませんでした、フォールバック応答を生成")
                # フォールバック応答を生成
                fallback_text = self._generate_fallback_text()
                self._response_generated = True
                self._response_complete = True
                
                if '/' in fallback_text:
                    phrase, emotion_action = fallback_text.rsplit('/', 1)
                    expression, action = _parse_split(emotion_action)
                    return {
                        'phrase': phrase.strip(),
                        'expression': expression,
                        'action': action
                    }
                else:
                    return {'phrase': fallback_text}
            
            # 応答終了
            self.log("応答ストリーム処理完了、正常終了")
            raise StopIteration
            
        except StopIteration:
            # StopIterationは正常な終了なので再発生
            self.log("応答生成正常終了")
            raise
        except Exception as e:
            self.log(f"応答生成エラー: {e}")
            import traceback
            traceback.print_exc()
            # エラーフラグを設定して無限ループを防ぐ
            self._error_occurred = True
            # エラー時は一度だけフォールバック応答を返して終了
            return {
                'phrase': '申し訳ございませんが、システムエラーが発生しました。',
                'expression': MMDAgentEXLabel.id2expression.get(4, 'normal'),
                'action': MMDAgentEXLabel.id2action.get(3, 'wait')
            }

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
        
        self.config = config
        self.prompts = prompts
        self.rag_retriever = rag_retriever
        
        # RAG応答統計
        self.usage_stats = {
            'total_queries': 0,
            'successful_responses': 0,
            'error_count': 0
        }
        
        # 互換性のための追加統計
        self.rag_usage_stats = {
            'total_requests': 0,
            'level_1_usage': 0,
            'level_2_usage': 0,
            'level_3_usage': 0
        }
        
        # RAG機能の設定
        self.model = config.get('ChatGPT', {}).get('response_generation_model', 'gpt-4.1-nano-2025-04-14')
        self.max_tokens = config.get('ChatGPT', {}).get('max_tokens', 128)
        self.max_message_num_in_context = config.get('ChatGPT', {}).get('max_message_num_in_context', 4)
        
        print(f"RAGResponseChatGPT初期化完了: モデル={self.model}, max_tokens={self.max_tokens}")
        print(f"設定確認 - response_generation_model: {config.get('ChatGPT', {}).get('response_generation_model')}")
    
    def run(self, asr_timestamp: float, user_utterance: Optional[str], 
            dialogue_history: List[Dict], last_asr_iu_id: Optional[str], 
            parent_llm_buffer):
        """RAG機能付き応答生成を実行"""
        try:
            self.usage_stats['total_queries'] += 1
            
            print(f"RAGResponseChatGPT実行開始: '{user_utterance}', モデル={self.model}")
            
            # 自身をDialogueモジュールが持つLLMバッファに追加（重要！）
            self.user_utterance = user_utterance
            self.asr_time = asr_timestamp
            
            # RAG応答生成器を作成
            rag_generator = RAGResponseGenerator(
                config=self.config,
                asr_timestamp=asr_timestamp,
                query=user_utterance,
                dialogue_history=dialogue_history,
                prompts=self.prompts,
                rag_retriever=self.rag_retriever
            )
            
            # 応答を生成してself.responseに設定
            response_parts = []
            for response_part in rag_generator:
                response_parts.append(response_part)
            
            # dialogue.pyが期待する形式で応答を設定
            self.response = iter(response_parts)
            
            # 成功統計を更新
            if response_parts:
                self.usage_stats['successful_responses'] += 1
                print(f"RAG応答生成完了: {len(response_parts)}個の応答部分を生成")
            else:
                print(f"[WARNING] 応答部分が生成されませんでした")
            
            # 自身をバッファに追加
            parent_llm_buffer.put(self)
            
        except Exception as e:
            self.usage_stats['error_count'] += 1
            print(f"RAGResponseChatGPT実行エラー: {e}")
            import traceback
            traceback.print_exc()
            
            # エラー時のフォールバック応答
            self.user_utterance = user_utterance or ''
            self.asr_time = asr_timestamp
            fallback_response = self._create_fallback_response(asr_timestamp, user_utterance, dialogue_history)
            self.response = iter([fallback_response])
            
            # エラー時もバッファに追加
            parent_llm_buffer.put(self)
    
    def _create_fallback_response(self, asr_timestamp: float, user_utterance: Optional[str], 
                                dialogue_history: List[Dict]):
        """フォールバック応答を作成"""
        if user_utterance:
            fallback_content = f"申し訳ございませんが、'{user_utterance}'についてお答えできませんでした。他にご質問はありますか？"
        else:
            fallback_content = "システムエラーが発生しました。申し訳ございません。"
        
        return {
            'phrase': fallback_content,
            'expression': MMDAgentEXLabel.id2expression.get(4, 'normal'),  # 考え中
            'action': MMDAgentEXLabel.id2action.get(3, 'wait')  # 首をかしげる
        }
    
    def _update_usage_stats(self):
        """使用統計を更新"""
        total = self.usage_stats['total_queries']
        success = self.usage_stats['successful_responses']
        errors = self.usage_stats['error_count']
        
        if total > 0:
            success_rate = (success / total) * 100
            print(f"RAG応答統計: 総クエリ数={total}, 成功={success}, エラー={errors}, 成功率={success_rate:.1f}%")
    
    def get_rag_stats(self) -> Dict[str, Any]:
        """RAG統計情報を取得"""
        return {
            **self.usage_stats,
            'success_rate': (self.usage_stats['successful_responses'] / max(1, self.usage_stats['total_queries'])) * 100,
            'model': self.model,
            'max_tokens': self.max_tokens
        }


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
    