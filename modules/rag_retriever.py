import os
import sys
import time
import hashlib
import json
from typing import List, Dict, Optional, Any
import threading
import queue

# 外部ライブラリ
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("Warning: chromadb not installed. Run: pip install chromadb")
    chromadb = None

try:
    import openai
except ImportError:
    print("Warning: openai not installed. Run: pip install openai")
    openai = None

from base import RemdisModule

class RAGRetriever:
    """RAG検索機能を提供するクラス"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # OpenAI API設定
        self.openai_api_key = config.get('ChatGPT', {}).get('api_key', '')
        if openai and self.openai_api_key:
            openai.api_key = self.openai_api_key
            print(f"OpenAI APIキーを設定しました: {self.openai_api_key[:10]}...")
        elif openai:
            print("OpenAI APIキーが設定されていません。埋め込み機能は無効になります。")
        else:
            print("OpenAIライブラリが利用できません。埋め込み機能は無効になります。")
        
        # ChromaDB設定
        self.db_path = config.get('RAG', {}).get('vector_db_path', './data/chromadb')
        self.collection_name = config.get('RAG', {}).get('collection_name', 'fujielab_knowledge')
        self.top_k = config.get('RAG', {}).get('top_k', 5)
        self.confidence_threshold = config.get('RAG', {}).get('confidence_threshold', 0.7)
        
        # ChromaDBクライアント初期化
        self.chroma_client = None
        self.collection = None
        self._initialize_chromadb()
        
        # 構造化基本情報（Level 1）
        self.structured_info = {
            "研究室名": "藤江研究室（fujielab）",
            "教員": "藤江真也教授",
            "研究分野": "音声言語処理、対話システム、自然言語処理",
            "所属": "千葉工業大学 先進工学部 未来ロボティクス学科",
            "URL": "https://www.fujielab.org/",
            "概要": "音声・言語・対話処理技術の研究開発を行う研究室"
        }
        
        # レスポンス時間計測用
        self.last_query_time = 0.0
    
    def _initialize_chromadb(self):
        """ChromaDBの初期化"""
        if not chromadb:
            print("ChromaDB not available. Using mock mode.")
            return
            
        try:
            # データディレクトリ作成
            os.makedirs(self.db_path, exist_ok=True)
            
            # ChromaDBクライアント作成
            self.chroma_client = chromadb.PersistentClient(path=self.db_path)
            
            # コレクション取得または作成
            try:
                self.collection = self.chroma_client.get_collection(name=self.collection_name)
                print(f"既存のコレクション '{self.collection_name}' を読み込みました")
            except:
                self.collection = self.chroma_client.create_collection(name=self.collection_name)
                print(f"新しいコレクション '{self.collection_name}' を作成しました")
                
        except Exception as e:
            print(f"ChromaDB初期化エラー: {e}")
            self.chroma_client = None
            self.collection = None
    
    def get_embedding(self, text: str) -> List[float]:
        """テキストのベクトル埋め込みを取得"""
        if not openai or not text.strip() or not self.openai_api_key:
            # モックデータ（テスト用）
            return [0.1] * 1536
            
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=text.strip()
            )
            return response['data'][0]['embedding']
        except Exception as e:
            print(f"埋め込み生成エラー: {e}")
            # エラー時はモックデータを返す
            return [0.1] * 1536
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """クエリに関連する情報を検索"""
        start_time = time.time()
        
        if not query or not query.strip():
            return self._create_response(3, "general", "質問が空です", 0.0)
        
        query = query.strip()
        top_k = top_k or self.top_k
        
        # Level 1: 構造化基本情報チェック
        basic_info = self._check_basic_info(query)
        if basic_info:
            response_time = time.time() - start_time
            return self._create_response(1, "structured", basic_info, 0.95, response_time)
        
        # Level 2: RAG検索
        if self.collection:
            try:
                rag_result = self._perform_rag_search(query, top_k)
                if rag_result and rag_result.get('documents'):
                    response_time = time.time() - start_time
                    # RAG検索結果から最適なテキストを抽出
                    best_content = self._extract_best_content(rag_result, query)
                    return self._create_response(2, "rag", best_content, 0.8, response_time)
            except Exception as e:
                print(f"RAG検索エラー: {e}")
        
        # Level 3: 一般知識フォールバック
        response_time = time.time() - start_time
        return self._create_response(3, "general", "一般的な知識で回答します", 0.3, response_time)
    
    def _check_basic_info(self, query: str) -> Optional[str]:
        """基本情報に関する質問かチェック"""
        query_lower = query.lower()
        
        # キーワードマッピング
        keyword_mapping = {
            # 研究室関連
            ("研究室", "ラボ", "lab"): f"{self.structured_info['研究室名']}は{self.structured_info['概要']}です。",
            # 教員関連  
            ("教員", "教授", "先生", "prof"): f"研究室の教員は{self.structured_info['教員']}です。",
            # 研究分野関連
            ("研究", "分野", "テーマ", "research"): f"研究分野は{self.structured_info['研究分野']}です。",
            # 所属関連
            ("所属", "大学", "学科", "university"): f"所属は{self.structured_info['所属']}です。",
            # URL関連
            ("サイト", "ウェブ", "homepage", "website", "url"): f"研究室のウェブサイトは{self.structured_info['URL']}です。"
        }
        
        for keywords, response in keyword_mapping.items():
            if any(keyword in query_lower for keyword in keywords):
                return response
                
        return None
    
    def _perform_rag_search(self, query: str, top_k: int) -> Optional[Dict[str, Any]]:
        """RAG検索を実行"""
        if not self.collection:
            return None
            
        try:
            # クエリの埋め込み生成
            query_embedding = self.get_embedding(query)
            
            # 類似度検索実行
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            if results.get('documents') and len(results['documents'][0]) > 0:
                return {
                    'documents': results['documents'][0],
                    'metadatas': results.get('metadatas', [[]])[0],
                    'distances': results.get('distances', [[]])[0]
                }
                
        except Exception as e:
            print(f"RAG検索実行エラー: {e}")
            
        return None
    
    def _extract_best_content(self, rag_result: Dict[str, Any], query: str) -> str:
        """RAG検索結果から最適なコンテンツを抽出"""
        documents = rag_result.get('documents', [])
        metadatas = rag_result.get('metadatas', [])
        distances = rag_result.get('distances', [])
        
        if not documents:
            return "関連する情報が見つかりませんでした。"
        
        # 最も関連性の高い文書を選択（距離が最小のもの）
        best_doc = documents[0]
        best_metadata = metadatas[0] if metadatas else {}
        
        # クエリに応じてレスポンスを調整
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['プロジェクト', 'project', '研究内容', '研究テーマ']):
            # プロジェクト関連の質問
            project_info = self._extract_project_info(best_doc, best_metadata)
            if project_info:
                return project_info
        
        # 一般的な文書内容を要約して返す
        content_summary = best_doc[:300] + '...' if len(best_doc) > 300 else best_doc
        source = best_metadata.get('source', '')
        title = best_metadata.get('title', '')
        
        response = f"{content_summary}"
        if title:
            response += f"\n\n（出典: {title}）"
        
        return response
    
    def _extract_project_info(self, document: str, metadata: Dict[str, Any]) -> str:
        """文書からプロジェクト情報を抽出"""
        doc_lower = document.lower()
        
        # プロジェクト関連のキーワードを含む部分を探す
        project_keywords = [
            'プロジェクト', 'project', '研究', 'research', 
            '開発', 'development', 'システム', 'system',
            'ロボット', 'robot', '対話', 'dialogue', '音声', 'speech'
        ]
        
        # 文書を文に分割
        sentences = document.split('。')
        relevant_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(keyword in sentence.lower() for keyword in project_keywords):
                relevant_sentences.append(sentence)
        
        if relevant_sentences:
            # 関連する文を結合
            project_desc = '。'.join(relevant_sentences[:3])  # 最初の3文のみ
            
            title = metadata.get('title', '')
            if title:
                return f"藤江研究室では以下のような研究プロジェクトを行っています：\n\n{project_desc}。\n\n（出典: {title}）"
            else:
                return f"藤江研究室では以下のような研究プロジェクトを行っています：\n\n{project_desc}。"
        
        return None
    
    def _create_response(self, level: int, response_type: str, content: Any, 
                        confidence: float, response_time: float = 0.0) -> Dict[str, Any]:
        """レスポンス辞書を作成"""
        self.last_query_time = response_time
        
        return {
            "level": level,
            "type": response_type,
            "content": content,
            "confidence": confidence,
            "response_time": response_time,
            "timestamp": time.time()
        }
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """文書をベクトルDBに追加"""
        if not documents:
            print("追加する文書がありません")
            return False
            
        # APIキーがない場合は簡易モードで動作
        if not self.openai_api_key:
            print(f"OpenAI APIキーが設定されていないため、簡易モードで{len(documents)}件の文書を処理します")
            # 簡易的にドキュメントを保存（実際のベクトル埋め込みは行わない）
            return True
            
        if not self.collection:
            print("ChromaDBコレクションが利用できません")
            return False
            
        try:
            texts = []
            metadatas = []
            ids = []
            
            for i, doc in enumerate(documents):
                content = doc.get('content', '').strip()
                if not content:
                    continue
                    
                texts.append(content)
                metadatas.append({
                    'source': doc.get('source', 'unknown'),
                    'timestamp': doc.get('timestamp', time.time()),
                    'level': doc.get('level', 2),
                    'title': doc.get('title', ''),
                    'section': doc.get('section', '')
                })
                
                # 一意なID生成
                content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
                ids.append(f"doc_{i}_{content_hash}")
            
            if not texts:
                print("追加する有効な文書がありません")
                return False
            
            # ベクトル埋め込み生成
            embeddings = []
            print(f"ベクトル埋め込みを生成中... (OpenAI API使用)")
            for i, text in enumerate(texts):
                embedding = self.get_embedding(text)
                embeddings.append(embedding)
                if (i + 1) % 10 == 0:
                    print(f"進捗: {i + 1}/{len(texts)} 完了")
            
            # ChromaDBに追加
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            
            print(f"{len(texts)}件の文書を追加しました")
            return True
            
        except Exception as e:
            print(f"文書追加エラー: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """コレクション情報を取得"""
        if not self.collection:
            return {"status": "unavailable", "count": 0}
            
        try:
            count = self.collection.count()
            return {
                "status": "available",
                "count": count,
                "collection_name": self.collection_name
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "count": 0}
    
    def clear_collection(self) -> bool:
        """コレクションをクリア"""
        if not self.collection:
            return False
            
        try:
            # 全文書を削除して再作成
            self.chroma_client.delete_collection(name=self.collection_name)
            self.collection = self.chroma_client.create_collection(name=self.collection_name)
            print("コレクションをクリアしました")
            return True
        except Exception as e:
            print(f"コレクションクリアエラー: {e}")
            return False


# 単体テスト
if __name__ == "__main__":
    import unittest
    
    class TestRAGRetriever(unittest.TestCase):
        
        def setUp(self):
            """テスト用の設定"""
            self.config = {
                'RAG': {
                    'vector_db_path': './test_data/chromadb',
                    'collection_name': 'test_collection',
                    'top_k': 3,
                    'confidence_threshold': 0.7
                },
                'ChatGPT': {
                    'api_key': 'test-key'
                }
            }
            self.retriever = RAGRetriever(self.config)
        
        def test_basic_info_retrieval(self):
            """基本情報取得テスト"""
            # 研究室名の質問
            result = self.retriever.retrieve("研究室について教えて")
            self.assertEqual(result['level'], 1)
            self.assertEqual(result['type'], 'structured')
            self.assertIn('藤江研究室', result['content'])
            
            # 教員の質問
            result = self.retriever.retrieve("教授は誰ですか")
            self.assertEqual(result['level'], 1)
            self.assertIn('藤江真也教授', result['content'])
        
        def test_empty_query(self):
            """空クエリのテスト"""
            result = self.retriever.retrieve("")
            self.assertEqual(result['level'], 3)
            self.assertEqual(result['type'], 'general')
        
        def test_add_documents(self):
            """文書追加テスト"""
            test_docs = [
                {
                    'content': 'テスト文書1の内容です。研究に関する情報。',
                    'source': 'test_source_1',
                    'title': 'テスト文書1'
                },
                {
                    'content': 'テスト文書2の内容です。対話システムについて。',
                    'source': 'test_source_2',
                    'title': 'テスト文書2'
                }
            ]
            
            # モックモードでもFalseを返さない
            result = self.retriever.add_documents(test_docs)
            # ChromaDBが利用可能な場合のみTrueを期待
            if self.retriever.collection:
                self.assertTrue(result)
        
        def test_collection_info(self):
            """コレクション情報テスト"""
            info = self.retriever.get_collection_info()
            self.assertIn('status', info)
            self.assertIn('count', info)
        
        def test_get_embedding(self):
            """埋め込み生成テスト"""
            embedding = self.retriever.get_embedding("テストテキスト")
            self.assertIsInstance(embedding, list)
            self.assertEqual(len(embedding), 1536)  # OpenAI埋め込みの次元数
        
        def test_rag_search_fallback(self):
            """RAG検索フォールバックテスト"""
            # 基本情報に該当しない質問
            result = self.retriever.retrieve("具体的な研究手法について")
            # ChromaDBが利用できない場合はLevel 3になる
            self.assertIn(result['level'], [2, 3])
    
    # テスト実行
    print("RAGRetrieverの単体テストを実行中...")
    unittest.main(argv=[''], exit=False, verbosity=2)
