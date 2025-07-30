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

try:
    from rag_llm import RAGResponseChatGPT
except ImportError:
    print("Warning: RAGResponseChatGPT not available")
    RAGResponseChatGPT = None


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
            prompt_dict = self.config.get('ChatGPT', {}).get('prompts', {})
            print(f"[DEBUG] 設定からのプロンプト辞書: {prompt_dict}")
            if prompt_dict:
                self.prompts = self._load_prompts(prompt_dict)
                print(f"[DEBUG] プロンプト設定: {list(self.prompts.keys())}")
                print(f"[DEBUG] RESPプロンプトの長さ: {len(self.prompts.get('RESP', ''))}")
            else:
                self.prompts = {'RESP': '30文字以内で簡潔に回答してください。', 'TO': 'ご質問はありますか？'}
                print("[DEBUG] デフォルトプロンプトを使用")
        except Exception as e:
            print(f"[DEBUG] プロンプト読み込みエラー: {e}")
            import traceback
            traceback.print_exc()
            self.prompts = {'RESP': '30文字以内で簡潔に回答してください。', 'TO': 'ご質問はありますか？'}
        
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
        
        # RAG関連の属性を初期化
        self.rag_response_chatgpt = None
        
        # RAGコンポーネントの初期化
        if self.rag_enabled:
            print("[DEBUG] RAGが有効化されています。コンポーネント初期化を開始...")
            self._initialize_rag_components()
        else:
            print("[DEBUG] RAGが無効化されています")
    
    def _load_prompts(self, prompt_dict):
        """設定ファイルからプロンプトを読み込む"""
        import os
        prompts = {}
        
        print(f"[DEBUG] _load_prompts開始: {prompt_dict}")
        
        for key, filepath in prompt_dict.items():
            try:
                print(f"[DEBUG] 処理中のプロンプト: {key} = {filepath}")
                
                # ファイルパスを解決
                if not os.path.isabs(filepath):
                    # 相対パスの場合、modulesディレクトリから見た相対パス
                    module_dir = os.path.dirname(__file__)
                    filepath = os.path.join(module_dir, filepath)
                
                print(f"[DEBUG] 解決されたパス: {filepath}")
                
                # ファイルが存在するかチェック
                if os.path.exists(filepath):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        prompts[key] = content
                    print(f"プロンプト '{key}' を読み込みました: {filepath} (長さ: {len(content)})")
                else:
                    print(f"プロンプトファイルが見つかりません: {filepath}")
                    prompts[key] = self._get_default_prompt(key)
                    print(f"[DEBUG] デフォルトプロンプトを使用: {key}")
            except Exception as e:
                print(f"プロンプト '{key}' の読み込みに失敗: {e}")
                import traceback
                traceback.print_exc()
                prompts[key] = self._get_default_prompt(key)
        
        print(f"[DEBUG] _load_prompts完了: {list(prompts.keys())}")
        return prompts
    
    def _get_default_prompt(self, key):
        """デフォルトプロンプトを取得"""
        defaults = {
            'RESP': 'あなたは藤江研究室のAIです。30文字以内で簡潔に回答してください。',
            'TO': 'ご質問をどうぞ。',
            'BC': 'うなずいてください。'
        }
        return defaults.get(key, '簡潔に回答してください。')
        
    def _initialize_rag_components(self):
        """RAG関連コンポーネントを初期化"""
        # RAG用ChatGPT応答生成器の初期化
        try:
            print(f"[DEBUG] RAGResponseChatGPT初期化開始: {RAGResponseChatGPT is not None}")
            if RAGResponseChatGPT:
                print(f"[DEBUG] RAGResponseChatGPT初期化パラメータ:")
                print(f"  - config: {type(self.config)}")
                print(f"  - prompts: {type(self.prompts)}")
                print(f"  - rag_retriever: {type(self.rag_retriever)}")
                
                self.rag_response_chatgpt = RAGResponseChatGPT(self.config, self.prompts, self.rag_retriever)
                print("RAG用ChatGPT応答生成器を初期化しました")
            else:
                print("RAGResponseChatGPTクラスが利用できません")
                self.rag_response_chatgpt = None
        except Exception as e:
            print(f"RAG用ChatGPT応答生成器の初期化に失敗: {e}")
            import traceback
            traceback.print_exc()
            self.rag_response_chatgpt = None
        
        # RAG機能の有効/無効チェック
        try:
            if not self.rag_enabled:
                print("RAG機能は無効化されています")
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
        
        self._is_running = True
        print("RAG対話Remdisモジュールを初期化しました")
    
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
                    # RAGResponseChatGPTが利用可能な場合はChatGPT経由で応答生成
                    if hasattr(self, 'rag_response_chatgpt') and self.rag_response_chatgpt:
                        try:
                            print(f"[DEBUG] RAGResponseChatGPT実行開始: '{user_utt}'")
                            print(f"[DEBUG] プロンプト設定: {list(self.prompts.keys())}")
                            print(f"[DEBUG] ChatGPTモデル: {self.config.get('ChatGPT', {}).get('response_generation_model', 'unknown')}")
                            
                            # RAG応答生成器を作成して応答を処理
                            from rag_llm import RAGResponseGenerator
                            print(f"[DEBUG] RAGResponseGenerator作成中...")
                            rag_generator = RAGResponseGenerator(
                                config=self.config,
                                asr_timestamp=time.time(),
                                query=user_utt,
                                dialogue_history=self.dialogue_history[-self.history_length:],
                                prompts=self.prompts,
                                rag_retriever=self.rag_retriever
                            )
                            print(f"[DEBUG] RAGResponseGenerator作成完了")
                            
                            # 応答を生成して送信
                            response_parts = []
                            full_response = ""
                            response_count = 0
                            max_response_parts = 20  # 無限ループ防止
                            
                            print(f"[DEBUG] 応答生成開始...")
                            for response_part in rag_generator:
                                response_count += 1
                                if response_count > max_response_parts:
                                    print(f"[WARNING] 応答部分数が上限({max_response_parts})に達しました。応答を終了します。")
                                    break
                                
                                response_parts.append(response_part)
                                print(f"[DEBUG] 応答パート受信: {response_part}")
                                
                                # 応答フラグメントを送信
                                if 'phrase' in response_part:
                                    phrase = response_part['phrase']
                                    if phrase.strip():
                                        full_response += phrase + " "
                                        snd_iu = self.createIU(phrase, 'dialogue', RemdisUpdateType.ADD)
                                        print('OUT(dialogue):', end='')
                                        self.printIU(snd_iu, flush=True)
                                        self.publish(snd_iu, 'dialogue')
                                        
                                        # 応答完了のチェック
                                        if any(end_marker in phrase for end_marker in ['。', '！', '？', '.']):
                                            print(f"[DEBUG] 応答完了マーカーを検出: {phrase}")
                                            break
                            
                            # 対話履歴を更新
                            if full_response.strip():
                                self._update_dialogue_history(user_utt, full_response.strip())
                            
                            print(f"[DEBUG] RAG応答生成完了: {len(response_parts)}個の応答部分を生成")
                            
                        except Exception as e:
                            print(f"[ERROR] RAGResponseChatGPT実行エラー: {e}")
                            import traceback
                            traceback.print_exc()
                            # フォールバック: 直接RAG検索結果を使用
                            print(f"[DEBUG] フォールバック処理に移行")
                            self._fallback_rag_response(user_utt)
                            return  # エラー後は処理を終了
                    else:
                        print(f"[DEBUG] RAGResponseChatGPTが利用できません - フォールバック処理")
                        # RAGResponseChatGPTが利用できない場合はフォールバック
                        self._fallback_rag_response(user_utt)
                else:
                    # 空の発話の場合
                    resp = "ご質問をどうぞ。"
                    snd_iu = self.createIU(resp, 'dialogue', RemdisUpdateType.ADD)
                    print('OUT(dialogue):', end='')
                    self.printIU(snd_iu, flush=True)
                    self.publish(snd_iu, 'dialogue')
                
                # バッファクリア
                self.user_utterance_buffer = []
            else:
                # バッファが空の場合
                resp = "ご質問をどうぞ。"
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
    
    def _fallback_rag_response(self, user_utt: str):
        """フォールバック用のRAG応答生成（簡易ChatGPT統合付き）"""
        try:
            print(f"[DEBUG] フォールバック処理開始: '{user_utt}'")
            
            # RAG検索実行
            start_time = time.time()
            rag_result = self.rag_retriever.retrieve(user_utt)
            response_time = time.time() - start_time
            print(f"[DEBUG] RAG検索結果: {rag_result}")
            
            # 統計更新
            self._update_rag_stats(rag_result, response_time)
            
            # RAG情報を取得
            rag_content = ""
            if isinstance(rag_result, dict):
                rag_content = rag_result.get('content', '')
            else:
                rag_content = str(rag_result)
            
            if not rag_content or rag_content.strip() == '':
                rag_content = f"「{user_utt}」について調べましたが、具体的な情報が見つかりませんでした。"
            
            print(f"[DEBUG] RAG情報: {rag_content[:100]}...")
            
            # システムプロンプトの作成
            system_prompt = f"""あなたは藤江研究室のAIです。30文字以内で簡潔に回答してください。

参考情報: {rag_content[:100]}
質問: {user_utt}

要求:
1. 30文字以内で回答
2. 簡潔で分かりやすく
3. 敬語は最小限に"""
            
            # 簡易ChatGPT統合を試行
            try:
                import openai
                api_key = self.config.get('ChatGPT', {}).get('api_key')
                if api_key:
                    print(f"[DEBUG] OpenAI API使用してRAG情報をChatGPTで処理")
                    
                    # OpenAI APIライブラリのバージョンに応じて適切な方法を使用
                    try:
                        # 新しいAPI (v1.0+) を試行
                        if hasattr(openai, 'OpenAI'):
                            client = openai.OpenAI(api_key=api_key)
                            
                            response = client.chat.completions.create(
                                model=self.config.get('ChatGPT', {}).get('response_generation_model', 'gpt-4.1-nano-2025-04-14'),
                                messages=[
                                    {"role": "system", "content": system_prompt}
                                ],
                                max_tokens=self.config.get('ChatGPT', {}).get('max_tokens', 32),
                                temperature=0.7
                            )
                            
                            resp = response.choices[0].message.content.strip()
                            print(f"[DEBUG] ChatGPT応答 (新API): {resp}")
                        else:
                            # 古いAPI (v0.x) を使用
                            openai.api_key = api_key
                            
                            response = openai.ChatCompletion.create(
                                model=self.config.get('ChatGPT', {}).get('response_generation_model', 'gpt-4.1-nano-2025-04-14'),
                                messages=[
                                    {"role": "system", "content": system_prompt}
                                ],
                                max_tokens=self.config.get('ChatGPT', {}).get('max_tokens', 32),
                                temperature=0.7
                            )
                            
                            resp = response.choices[0].message.content.strip()
                            print(f"[DEBUG] ChatGPT応答 (旧API): {resp}")
                    except Exception as api_error:
                        print(f"[DEBUG] API呼び出しエラー: {api_error}")
                        raise api_error
                else:
                    raise Exception("OpenAI APIキーが設定されていません")
                    
            except Exception as chatgpt_error:
                print(f"[DEBUG] ChatGPT処理失敗: {chatgpt_error}")
                # ChatGPTが失敗した場合、簡潔な応答を生成（32文字以内）
                if "こんにちは" in user_utt or "hello" in user_utt.lower():
                    resp = "こんにちは！藤江研究室です。"
                elif rag_content and len(rag_content.strip()) > 0:
                    # RAG情報を要約して簡潔にする（32文字以内）
                    content_summary = rag_content[:20].replace('\n', ' ').strip()
                    if content_summary:
                        resp = f"{content_summary}について研究しています。"[:30]
                    else:
                        resp = "音声対話技術を研究しています。"
                else:
                    resp = "申し訳ございません。"
            
            # 応答送信
            snd_iu = self.createIU(resp, 'dialogue', RemdisUpdateType.ADD)
            print('OUT(dialogue):', end='')
            self.printIU(snd_iu, flush=True)
            self.publish(snd_iu, 'dialogue')
            
            # 対話履歴を更新
            self._update_dialogue_history(user_utt, resp)
            
        except Exception as e:
            print(f"[ERROR] フォールバック応答生成エラー: {e}")
            import traceback
            traceback.print_exc()
            # 最終フォールバック
            resp = "申し訳ございませんが、システムエラーが発生しました。"
            snd_iu = self.createIU(resp, 'dialogue', RemdisUpdateType.ADD)
            print('OUT(dialogue):', end='')
            self.printIU(snd_iu, flush=True)
            self.publish(snd_iu, 'dialogue')
    
    def _update_dialogue_history(self, user_utterance: str, system_response: str):
        """対話履歴を更新"""
        # ユーザー発話を追加
        self.dialogue_history.append({
            'role': 'user',
            'content': user_utterance,
            'timestamp': time.time()
        })
        
        # システム応答を追加
        self.dialogue_history.append({
            'role': 'assistant',
            'content': system_response,
            'timestamp': time.time()
        })
        
        # 履歴長制限
        max_history = self.history_length * 2  # ユーザーとシステムのペアで計算
        if len(self.dialogue_history) > max_history:
            self.dialogue_history = self.dialogue_history[-max_history:]
    
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
            print(f"処理する発話: '{user_utterance}'")
            
            # RAGResponseChatGPTが利用可能な場合
            if hasattr(self, 'rag_response_chatgpt') and self.rag_response_chatgpt:
                # RAGResponseChatGPTを使用してChatGPT経由で応答生成
                try:
                    self.rag_response_chatgpt.run(
                        asr_timestamp=time.time(),
                        user_utterance=user_utterance,
                        dialogue_history=self.dialogue_history[-self.history_length:],
                        last_asr_iu_id=input_iu.get('id'),
                        parent_llm_buffer=self.llm_buffer
                    )
                    return
                except Exception as e:
                    print(f"RAGResponseChatGPT実行エラー: {e}")
                    # フォールバックに進む
            
            # フォールバック: 直接RAG検索結果を使用
            if self.rag_enabled and self.rag_retriever:
                start_time = time.time()
                rag_result = self.rag_retriever.retrieve(user_utterance)
                response_time = time.time() - start_time
                
                # 統計更新
                self._update_rag_stats(rag_result, response_time)
                
                # レスポンス内容を取得
                if rag_result and rag_result.get('content'):
                    response_content = rag_result['content']
                else:
                    response_content = "申し訳ございませんが、関連する情報が見つかりませんでした。"
                
                # 即座に応答を送信
                response_iu = self.createIU(response_content, 'dialogue', RemdisUpdateType.ADD)
                print('OUT(dialogue):', end='')
                self.printIU(response_iu, flush=True)
                self.publish(response_iu, 'dialogue')
                
                # デバッグ情報送信
                self._send_rag_debug_info(user_utterance, rag_result)
            else:
                # RAGが無効な場合のフォールバック
                fallback_response = "こんにちは、藤江研究室です。お手伝いできることがあれば教えてください。"
                response_iu = self.createIU(fallback_response, 'dialogue', RemdisUpdateType.ADD)
                print('OUT(dialogue):', end='')
                self.printIU(response_iu, flush=True)
                self.publish(response_iu, 'dialogue')
            
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
                print(f"[DEBUG] config.yaml読み込み後のChatGPT設定: {config.get('ChatGPT', {})}")
        
        # API設定ファイルの読み込み
        if os.path.exists(api_config_path):
            import yaml
            with open(api_config_path, 'r', encoding='utf-8') as f:
                api_config = yaml.safe_load(f) or {}
                print(f"[DEBUG] api_config.yaml内容: {api_config}")
                
                # API設定を適切にマージ（重要な設定を保護）
                for key, value in api_config.items():
                    if key in config:
                        if isinstance(config[key], dict) and isinstance(value, dict):
                            # 辞書の場合は既存の設定を保護してマージ
                            for sub_key, sub_value in value.items():
                                config[key][sub_key] = sub_value
                        else:
                            config[key] = value
                    else:
                        config[key] = value
                
                print(f"[DEBUG] 保護的マージ後のChatGPT設定: {config.get('ChatGPT', {})}")
        else:
            print("[DEBUG] api_config.yamlが存在しません")
        
        # デフォルト設定の補完
        if 'RAG' not in config:
            config['RAG'] = {'enable': True}
        if 'DIALOGUE' not in config:
            config['DIALOGUE'] = {'history_length': 10}
        if 'ChatGPT' not in config:
            config['ChatGPT'] = {
                'prompts': {
                    'RESP': 'prompt/rag_system.txt',
                    'TO': 'prompt/time_out.txt'
                },
                'max_tokens': 32,
                'max_message_num_in_context': 4,
                'response_generation_model': 'gpt-4.1-nano-2025-04-14'
            }
        else:
            # ChatGPT設定が存在する場合、不足している項目を補完
            chatgpt_config = config['ChatGPT']
            if 'max_tokens' not in chatgpt_config:
                chatgpt_config['max_tokens'] = 32
            if 'max_message_num_in_context' not in chatgpt_config:
                chatgpt_config['max_message_num_in_context'] = 4
            if 'response_generation_model' not in chatgpt_config:
                chatgpt_config['response_generation_model'] = 'gpt-4.1-nano-2025-04-14'
        
        print(f"設定読み込み完了: ChatGPT APIキー = {'設定済み' if config.get('ChatGPT', {}).get('api_key') else '未設定'}")
        print(f"ChatGPT設定確認:")
        print(f"  - モデル: {config.get('ChatGPT', {}).get('response_generation_model', 'unknown')}")
        print(f"  - max_tokens: {config.get('ChatGPT', {}).get('max_tokens', 'unknown')}")
        print(f"  - max_message_num_in_context: {config.get('ChatGPT', {}).get('max_message_num_in_context', 'unknown')}")
        print(f"  - prompts: {config.get('ChatGPT', {}).get('prompts', {})}")
        
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
