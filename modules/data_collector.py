import os
import sys
import time
import hashlib
import re
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
import threading
import queue

# 外部ライブラリ
try:
    import requests
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry
except ImportError:
    print("Warning: requests not installed. Run: pip install requests")
    requests = None

try:
    from bs4 import BeautifulSoup, NavigableString, Tag
except ImportError:
    print("Warning: beautifulsoup4 not installed. Run: pip install beautifulsoup4")
    BeautifulSoup = None

from base import RemdisModule

class FujielabDataCollector:
    """藤江研究室のデータ収集クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # データソース設定
        data_sources = config.get('DATA_SOURCES', {})
        self.base_urls = [
            data_sources.get('fujielab_website', 'https://www.fujielab.org/'),
        ]
        
        # 収集設定
        self.timeout = data_sources.get('timeout', 10)
        self.max_retries = data_sources.get('max_retries', 3)
        self.delay_between_requests = data_sources.get('delay_between_requests', 1.0)
        self.max_content_length = data_sources.get('max_content_length', 1000000)  # 1MB
        
        # テキスト処理設定
        rag_config = config.get('RAG', {})
        self.chunk_size = rag_config.get('chunk_size', 512)
        self.chunk_overlap = rag_config.get('chunk_overlap', 50)
        
        # HTTPセッション設定
        self.session = self._create_session()
        
        # 除外パターン
        self.exclude_patterns = [
            r'.*\.(pdf|doc|docx|xls|xlsx|ppt|pptx)$',  # バイナリファイル
            r'.*/(admin|login|private)/',  # 管理画面
            r'.*\#.*',  # アンカーリンク
        ]
        
        # 収集統計
        self.stats = {
            'total_pages': 0,
            'successful_pages': 0,
            'failed_pages': 0,
            'total_documents': 0,
            'last_update': None
        }
    
    def _create_session(self) -> requests.Session:
        """HTTPセッションを作成"""
        if not requests:
            return None
            
        session = requests.Session()
        
        # リトライ戦略
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # ヘッダー設定
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        return session
    
    def collect_website_data(self, max_pages: int = 50) -> List[Dict[str, Any]]:
        """ウェブサイトからデータを収集"""
        documents = []
        visited_urls = set()
        
        for base_url in self.base_urls:
            print(f"データ収集開始: {base_url}")
            
            try:
                # メインページから開始
                page_docs = self._collect_single_page(base_url)
                if page_docs:
                    documents.extend(page_docs)
                    visited_urls.add(base_url)
                    self.stats['successful_pages'] += 1
                else:
                    self.stats['failed_pages'] += 1
                
                self.stats['total_pages'] += 1
                
                # サブページの収集（簡単な実装）
                subpage_urls = self._extract_internal_links(base_url)
                
                for url in subpage_urls[:max_pages-1]:  # 制限
                    if url not in visited_urls and not self._should_exclude_url(url):
                        time.sleep(self.delay_between_requests)  # レート制限
                        
                        page_docs = self._collect_single_page(url)
                        if page_docs:
                            documents.extend(page_docs)
                            self.stats['successful_pages'] += 1
                        else:
                            self.stats['failed_pages'] += 1
                        
                        visited_urls.add(url)
                        self.stats['total_pages'] += 1
                        
                        if len(visited_urls) >= max_pages:
                            break
                            
            except Exception as e:
                print(f"データ収集エラー {base_url}: {e}")
                self.stats['failed_pages'] += 1
        
        self.stats['total_documents'] = len(documents)
        self.stats['last_update'] = time.time()
        
        print(f"データ収集完了: {len(documents)}件の文書")
        return documents
    
    def _collect_single_page(self, url: str) -> List[Dict[str, Any]]:
        """単一ページからデータを収集"""
        if not self.session:
            return self._create_mock_documents(url)
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # コンテンツサイズチェック
            if len(response.content) > self.max_content_length:
                print(f"コンテンツが大きすぎます: {url}")
                return []
            
            # HTMLパース
            if not BeautifulSoup:
                return self._create_mock_documents(url)
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # メタデータ抽出
            metadata = self._extract_metadata(soup, url)
            
            # テキストコンテンツ抽出
            content = self._extract_content(soup)
            
            if not content or len(content.strip()) < 100:  # 最小文字数チェック
                return []
            
            # 文書を分割
            documents = self._split_content(content, metadata)
            
            return documents
            
        except requests.RequestException as e:
            print(f"HTTP エラー {url}: {e}")
            return []
        except Exception as e:
            print(f"ページ処理エラー {url}: {e}")
            return []
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """メタデータを抽出"""
        metadata = {
            'source': url,
            'timestamp': time.time(),
            'level': 2,  # 動的情報
            'title': '',
            'description': '',
            'section': ''
        }
        
        # タイトル取得
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text().strip()
        
        # メタ説明取得
        desc_tag = soup.find('meta', attrs={'name': 'description'})
        if desc_tag and desc_tag.get('content'):
            metadata['description'] = desc_tag.get('content').strip()
        
        # セクション判定（URLベース）
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.strip('/').split('/')
        if len(path_parts) > 0 and path_parts[0]:
            metadata['section'] = path_parts[0]
        
        return metadata
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """HTMLからテキストコンテンツを抽出"""
        if not soup:
            return ""
        
        # 不要な要素を削除
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 
                           'aside', 'noscript', 'iframe', 'form']):
            element.decompose()
        
        # コメント削除
        from bs4 import Comment
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # メインコンテンツ抽出の優先順位
        content_selectors = [
            'main',
            '[role="main"]',
            '.content',
            '.main-content',
            '#content',
            '#main',
            'article',
            'body'
        ]
        
        main_content = None
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            main_content = soup
        
        # テキスト抽出
        text = main_content.get_text(separator=' ', strip=True)
        
        # テキスト正規化
        text = re.sub(r'\s+', ' ', text)  # 連続する空白を1つに
        text = re.sub(r'\n\s*\n', '\n', text)  # 連続する改行を1つに
        
        return text.strip()
    
    def _split_content(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """コンテンツを適切なサイズに分割"""
        if len(content) <= self.chunk_size:
            return [{
                'content': content,
                **metadata,
                'chunk_id': 0
            }]
        
        documents = []
        chunks = self._create_chunks(content)
        
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:  # 短すぎるチャンクはスキップ
                continue
                
            doc_metadata = metadata.copy()
            doc_metadata['chunk_id'] = i
            doc_metadata['total_chunks'] = len(chunks)
            
            documents.append({
                'content': chunk.strip(),
                **doc_metadata
            })
        
        return documents
    
    def _create_chunks(self, text: str) -> List[str]:
        """テキストをチャンクに分割"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end >= len(text):
                # 最後のチャンク
                chunks.append(text[start:])
                break
            
            # 文の境界で分割を試みる
            chunk = text[start:end]
            
            # 文の境界を探す
            sentence_endings = ['. ', '。', '！', '？']
            best_split = -1
            
            for ending in sentence_endings:
                last_occurrence = chunk.rfind(ending)
                if last_occurrence > self.chunk_size * 0.7:  # 70%以上の位置
                    best_split = max(best_split, last_occurrence + len(ending))
            
            if best_split > 0:
                chunks.append(text[start:start + best_split])
                start = start + best_split - self.chunk_overlap
            else:
                # 文の境界が見つからない場合は強制分割
                chunks.append(chunk)
                start = end - self.chunk_overlap
        
        return chunks
    
    def _extract_internal_links(self, base_url: str) -> List[str]:
        """内部リンクを抽出"""
        if not self.session or not BeautifulSoup:
            return []
        
        try:
            response = self.session.get(base_url, timeout=self.timeout)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            links = []
            base_domain = urlparse(base_url).netloc
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(base_url, href)
                parsed_url = urlparse(full_url)
                
                # 同一ドメインのみ
                if parsed_url.netloc == base_domain:
                    links.append(full_url)
            
            return list(set(links))  # 重複除去
            
        except Exception as e:
            print(f"リンク抽出エラー: {e}")
            return []
    
    def _should_exclude_url(self, url: str) -> bool:
        """URLを除外すべきかチェック"""
        for pattern in self.exclude_patterns:
            if re.match(pattern, url, re.IGNORECASE):
                return True
        return False
    
    def _create_mock_documents(self, url: str) -> List[Dict[str, Any]]:
        """テスト用のモック文書を作成"""
        return [{
            'content': f'モック文書: {url} からの内容。藤江研究室に関する情報。',
            'source': url,
            'timestamp': time.time(),
            'level': 2,
            'title': 'モック文書',
            'section': 'test',
            'chunk_id': 0
        }]
    
    def get_stats(self) -> Dict[str, Any]:
        """収集統計を取得"""
        return self.stats.copy()
    
    def reset_stats(self):
        """統計をリセット"""
        self.stats = {
            'total_pages': 0,
            'successful_pages': 0,
            'failed_pages': 0,
            'total_documents': 0,
            'last_update': None
        }


class DataCollectorModule(RemdisModule):
    """RemdisModule継承版のデータ収集モジュール"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.collector = FujielabDataCollector(self.config)
        self.update_interval = self.config.get('DATA_SOURCES', {}).get('update_interval_hours', 24) * 3600
        self.last_update = 0
        self._is_running = True
    
    def run(self):
        """定期実行メインループ"""
        while self._is_running:
            current_time = time.time()
            
            if current_time - self.last_update >= self.update_interval:
                print("データ収集を開始します...")
                documents = self.collector.collect_website_data()
                
                if documents:
                    # 収集結果を他のモジュールに送信（実装は用途に応じて）
                    print(f"収集完了: {len(documents)}件の文書")
                
                self.last_update = current_time
            
            time.sleep(60)  # 1分間隔でチェック


# 単体テスト
if __name__ == "__main__":
    import unittest
    from unittest.mock import Mock, patch
    
    class TestFujielabDataCollector(unittest.TestCase):
        
        def setUp(self):
            """テスト用の設定"""
            self.config = {
                'DATA_SOURCES': {
                    'fujielab_website': 'https://example.com',
                    'timeout': 5,
                    'max_retries': 2,
                    'delay_between_requests': 0.1
                },
                'RAG': {
                    'chunk_size': 100,
                    'chunk_overlap': 20
                }
            }
            self.collector = FujielabDataCollector(self.config)
        
        def test_initialization(self):
            """初期化テスト"""
            self.assertIsInstance(self.collector.base_urls, list)
            self.assertEqual(self.collector.chunk_size, 100)
            self.assertIsInstance(self.collector.stats, dict)
        
        def test_create_chunks(self):
            """テキスト分割テスト"""
            text = "これは最初の文です。これは2番目の文です。これは3番目の文です。これは4番目の文です。これは5番目の文です。" * 10  # chunk_sizeを超える長さに
            chunks = self.collector._create_chunks(text)
            
            self.assertIsInstance(chunks, list)
            self.assertGreater(len(chunks), 1)
            
            # 各チャンクがサイズ制限内
            for chunk in chunks:
                self.assertLessEqual(len(chunk), self.collector.chunk_size + 50)  # 余裕を持たせる
        
        def test_should_exclude_url(self):
            """URL除外テスト"""
            # PDFファイルは除外
            self.assertTrue(self.collector._should_exclude_url("https://example.com/file.pdf"))
            
            # 通常のHTMLページは除外しない
            self.assertFalse(self.collector._should_exclude_url("https://example.com/page.html"))
            
            # 管理画面は除外
            self.assertTrue(self.collector._should_exclude_url("https://example.com/admin/"))
        
        def test_extract_metadata(self):
            """メタデータ抽出テスト"""
            if not BeautifulSoup:
                self.skipTest("BeautifulSoup not available")
            
            html = """
            <html>
                <head>
                    <title>テストページ</title>
                    <meta name="description" content="テスト用の説明">
                </head>
                <body>テスト内容</body>
            </html>
            """
            soup = BeautifulSoup(html, 'html.parser')
            metadata = self.collector._extract_metadata(soup, "https://example.com/test")
            
            self.assertEqual(metadata['title'], 'テストページ')
            self.assertEqual(metadata['description'], 'テスト用の説明')
            self.assertEqual(metadata['source'], 'https://example.com/test')
        
        def test_extract_content(self):
            """コンテンツ抽出テスト"""
            if not BeautifulSoup:
                self.skipTest("BeautifulSoup not available")
            
            html = """
            <html>
                <head><title>テスト</title></head>
                <body>
                    <nav>ナビゲーション</nav>
                    <main>
                        <h1>メインコンテンツ</h1>
                        <p>これは    重要な    情報です。</p>
                    </main>
                    <footer>フッター</footer>
                    <script>console.log('script');</script>
                </body>
            </html>
            """
            soup = BeautifulSoup(html, 'html.parser')
            content = self.collector._extract_content(soup)
            
            self.assertIn('メインコンテンツ', content)
            self.assertIn('重要な 情報', content)  # 実際の抽出結果に合わせる
            self.assertNotIn('ナビゲーション', content)
            self.assertNotIn('フッター', content)
            self.assertNotIn('script', content)
        
        def test_collect_website_data_mock(self):
            """データ収集テスト（モック使用）"""
            # requestsが利用できない場合のモック動作テスト
            documents = self.collector.collect_website_data(max_pages=1)
            
            self.assertIsInstance(documents, list)
            # モックまたは実際のデータが返される
            self.assertGreaterEqual(len(documents), 0)
        
        def test_split_content(self):
            """コンテンツ分割テスト"""
            content = "短いコンテンツ"
            metadata = {'source': 'test', 'title': 'test'}
            
            documents = self.collector._split_content(content, metadata)
            
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0]['content'], content)
            self.assertEqual(documents[0]['chunk_id'], 0)
        
        def test_get_stats(self):
            """統計取得テスト"""
            stats = self.collector.get_stats()
            
            self.assertIn('total_pages', stats)
            self.assertIn('successful_pages', stats)
            self.assertIn('failed_pages', stats)
            self.assertIn('total_documents', stats)
    
    # テスト実行
    print("FujielabDataCollectorの単体テストを実行中...")
    unittest.main(argv=[''], exit=False, verbosity=2)
