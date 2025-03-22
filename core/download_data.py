import wikipedia
import json
from pathlib import Path
from typing import List, Dict
import time
from tqdm import tqdm
import random
import logging
from bs4 import BeautifulSoup
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from dotenv import load_dotenv
from core.storage import StorageService

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure BeautifulSoup parser
wikipedia.wikipedia.BeautifulSoup = lambda *args, **kwargs: BeautifulSoup(*args, features="html.parser", **kwargs)

class WikipediaDownloader:
    def __init__(self, output_dir: str = "data/wikipedia", max_workers: int = 100):
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
        
        # Set Wikipedia language to English
        wikipedia.set_lang("en")
        
        # List of categories to sample from
        self.categories = [
            "Science", "Technology", "History", "Geography", "Arts",
            "Sports", "Politics", "Business", "Education", "Health"
        ]
        
        # Thread-safe queue and set
        self.article_queue = Queue()
        self.seen_titles = set()
        self.seen_titles_lock = threading.Lock()
    
    def _download_single_article(self) -> Dict:
        """Download a single article from Wikipedia."""
        while True:
            try:
                # Randomly select a category
                category = random.choice(self.categories)
                
                # Get random page title from Wikipedia
                random_title = wikipedia.random(pages=1)
                
                # Thread-safe check for seen titles
                with self.seen_titles_lock:
                    if random_title in self.seen_titles:
                        continue
                    self.seen_titles.add(random_title)
                
                # Get the full page object
                page = wikipedia.page(random_title)
                
                # Get the full content
                content = page.content
                
                # Clean the content
                content = self._clean_content(content)
                
                if len(content.split()) < 100:  # Skip very short articles
                    continue
                
                return {
                    "title": page.title,
                    "text": content,
                    "category": category
                }
                
            except wikipedia.exceptions.DisambiguationError:
                continue
            except wikipedia.exceptions.PageError:
                continue
            except Exception as e:
                logger.error(f"Error downloading article: {str(e)}")
                continue
    
    def _clean_content(self, content: str) -> str:
        """Clean Wikipedia content."""
        # Remove section headers
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            if not line.startswith('==') and not line.startswith('==='):
                cleaned_lines.append(line)
        
        # Join lines and clean up whitespace
        cleaned_content = ' '.join(cleaned_lines)
        cleaned_content = ' '.join(cleaned_content.split())
        
        return cleaned_content
    
    def _save_articles(self, articles: List[Dict], output_file: str):
        """Save articles either locally or to GCS based on TEST environment variable."""
        if self.is_test:
            # Save locally
            output_path = self.output_dir / output_file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(articles, f, indent=2, ensure_ascii=False)
            logger.info(f"Articles saved locally to {output_path}")
        else:
            try:
                # Save to GCS using StorageService
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
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            future_to_article = {
                executor.submit(self._download_single_article): i 
                for i in range(num_articles)
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=num_articles, desc="Downloading articles") as pbar:
                for future in as_completed(future_to_article):
                    try:
                        article = future.result()
                        articles.append(article)
                        pbar.update(1)
                        # Add a small delay to be nice to Wikipedia's servers
                        time.sleep(0.1)
                    except Exception as e:
                        logger.error(f"Error processing article: {str(e)}")
        
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
    parser.add_argument('--max_workers', type=int, default=4,
                      help='Number of worker threads')
    
    args = parser.parse_args()
    
    downloader = WikipediaDownloader(output_dir=args.output_dir, max_workers=args.max_workers)
    downloader.download_and_save(
        num_articles=args.num_articles,
        output_file=args.output_file
    )