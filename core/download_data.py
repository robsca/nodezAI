import requests
import json
from pathlib import Path
from typing import List, Dict, Optional
import time
from tqdm import tqdm
import random
import logging
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from dotenv import load_dotenv
from core.storage import StorageService
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WikipediaDownloader:
    def __init__(self, output_dir: str = "data/wikipedia", max_workers: int = 8):  # Increased default workers
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        
        # Initialize storage service if not in test mode
        self.is_test = os.getenv('TEST', 'false').lower() == 'true'
        if not self.is_test:
            try:
                self.storage_service = StorageService()
            except Exception as e:
                logger.error(f"Failed to initialize StorageService: {str(e)}")
                raise
        
        # List of categories to sample from
        self.categories = [
            "Science", "Technology", "History", "Geography", "Arts",
            "Sports", "Politics", "Business", "Education", "Health"
        ]
        
        # Thread-safe queue and set
        self.article_queue = Queue()
        self.seen_titles = set()
        self.seen_titles_lock = threading.Lock()
        
        # Configure multiple sessions for parallel requests
        self.sessions = [self._create_session() for _ in range(max_workers)]
        self.session_lock = threading.Lock()
        self.current_session = 0
        
        # Rate limiting - separate counters per thread
        self.request_counts = {i: 0 for i in range(max_workers)}
        self.last_request_times = {i: 0 for i in range(max_workers)}
        self.rate_limit_locks = {i: threading.Lock() for i in range(max_workers)}
        self.rate_limit_delay = 0.5  # Reduced delay between requests
    
    def _create_session(self):
        """Create a requests session with retry strategy"""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=100, pool_maxsize=100)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        return session
    
    def _get_session(self):
        """Get a session in a thread-safe way"""
        with self.session_lock:
            session = self.sessions[self.current_session]
            self.current_session = (self.current_session + 1) % self.max_workers
            return session, self.current_session
    
    def _rate_limit(self, thread_id: int):
        """Implement rate limiting per thread"""
        with self.rate_limit_locks[thread_id]:
            current_time = time.time()
            elapsed = current_time - self.last_request_times[thread_id]
            
            if elapsed < self.rate_limit_delay:
                sleep_time = self.rate_limit_delay - elapsed
                time.sleep(sleep_time)
            
            self.last_request_times[thread_id] = time.time()
            self.request_counts[thread_id] += 1
    
    def _make_request(self, url, params=None, allow_redirects=False):
        """Make a request with rate limiting and error handling"""
        session, thread_id = self._get_session()
        self._rate_limit(thread_id)
        
        try:
            response = session.get(url, params=params, timeout=10, allow_redirects=allow_redirects)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error: {str(e)}")
            raise
    
    def _get_random_title(self) -> Optional[str]:
        """Get a random Wikipedia article title using the API"""
        try:
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "format": "json",
                "list": "random",
                "rnnamespace": "0",
                "rnlimit": "1"
            }
            
            response = self._make_request(url, params)
            data = response.json()
            
            if "query" in data and "random" in data["query"] and len(data["query"]["random"]) > 0:
                title = data["query"]["random"][0]["title"]
                return title
            else:
                logger.warning("No random title found in API response")
                return None
        except Exception as e:
            logger.error(f"Error getting random title: {str(e)}")
            return None
    
    def _get_article_content(self, title: str) -> Optional[Dict]:
        """Get article content using the Wikipedia API"""
        try:
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts",
                "explaintext": True,
                "exsectionformat": "plain"
            }
            
            response = self._make_request(url, params)
            data = response.json()
            
            pages = data["query"]["pages"]
            page_id = list(pages.keys())[0]
            
            if page_id == "-1":
                logger.warning(f"Page not found: {title}")
                return None
            
            page_data = pages[page_id]
            
            if "extract" not in page_data:
                logger.warning(f"No content found for: {title}")
                return None
            
            category = random.choice(self.categories)
            
            return {
                "title": page_data["title"],
                "text": page_data["extract"],
                "category": category
            }
        except Exception as e:
            logger.error(f"Error getting article content for '{title}': {str(e)}")
            return None
    
    def _download_single_article(self) -> Optional[Dict]:
        """Download a single article from Wikipedia."""
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                title = self._get_random_title()
                if not title:
                    logger.warning(f"Failed to get random title (attempt {attempt+1}/{max_attempts})")
                    time.sleep(2 ** attempt)
                    continue
                
                with self.seen_titles_lock:
                    if title in self.seen_titles:
                        logger.debug(f"Title already seen: {title}, trying again")
                        continue
                    self.seen_titles.add(title)
                
                article = self._get_article_content(title)
                if not article:
                    logger.warning(f"Failed to get content for title '{title}' (attempt {attempt+1}/{max_attempts})")
                    time.sleep(2 ** attempt)
                    continue
                
                article["text"] = self._clean_content(article["text"])
                
                if len(article["text"].split()) < 100:
                    logger.debug(f"Article too short: {title}, trying again")
                    continue
                
                return article
                
            except Exception as e:
                logger.error(f"Error processing article (attempt {attempt+1}/{max_attempts}): {str(e)}")
                time.sleep(2 ** attempt)
        
        return None
    
    def _clean_content(self, content: str) -> str:
        """Clean Wikipedia content."""
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            if not line.startswith('==') and not line.startswith('==='):
                cleaned_lines.append(line)
        
        cleaned_content = ' '.join(cleaned_lines)
        cleaned_content = ' '.join(cleaned_content.split())
        
        return cleaned_content
    
    def _save_articles(self, articles: List[Dict], output_file: str):
        """Save articles either locally or to GCS based on TEST environment variable."""
        if self.is_test:
            output_path = self.output_dir / output_file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(articles, f, indent=2, ensure_ascii=False)
            logger.info(f"Articles saved locally to {output_path}")
        else:
            try:
                success = self.storage_service.save_data(articles, output_file)
                if success:
                    logger.info(f"Articles saved to GCS successfully")
                else:
                    raise Exception("Failed to save articles to GCS")
            except Exception as e:
                logger.error(f"Failed to save articles to GCS: {str(e)}")
                raise
    
    def get_random_articles(self, num_articles: int = 1000) -> List[Dict]:
        """Download random articles from Wikipedia using multiple threads."""
        articles = []
        successful_downloads = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_article = {
                executor.submit(self._download_single_article): i 
                for i in range(num_articles)
            }
            
            with tqdm(total=num_articles, desc="Downloading articles") as pbar:
                for future in as_completed(future_to_article):
                    try:
                        article = future.result()
                        if article:
                            articles.append(article)
                            successful_downloads += 1
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Error processing article: {str(e)}")
                        pbar.update(1)
        
        logger.info(f"Successfully downloaded {successful_downloads} out of {num_articles} articles")
        return articles
    
    def download_and_save(self, num_articles: int = 1000, output_file: str = "wikipedia_articles.json"):
        """Download articles and save them based on TEST environment variable."""
        articles = self.get_random_articles(num_articles)
        self._save_articles(articles, output_file)
        return articles

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download Wikipedia articles for training')
    parser.add_argument('--num_articles', type=int, default=1000,
                      help='Number of articles to download')
    parser.add_argument('--output_dir', type=str, default='data/wikipedia',
                      help='Directory to save the downloaded articles')
    parser.add_argument('--output_file', type=str, default='wikipedia_articles.json',
                      help='Name of the output JSON file')
    parser.add_argument('--max_workers', type=int, default=8,  # Increased default workers
                      help='Number of worker threads')
    
    args = parser.parse_args()
    
    downloader = WikipediaDownloader(output_dir=args.output_dir, max_workers=args.max_workers)
    downloader.download_and_save(
        num_articles=args.num_articles,
        output_file=args.output_file
    )