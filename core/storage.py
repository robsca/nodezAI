from google.cloud import storage
from google.oauth2 import service_account
import os
from dotenv import load_dotenv
import logging
import json

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

cert_dict = {
    "type": os.getenv("FIREBASE_TYPE"),
    "project_id": os.getenv("FIREBASE_PROJECT_ID"), 
    "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
    "private_key": os.getenv('FIREBASE_PRIVATE_KEY').replace('\\n', '\n') if os.getenv('FIREBASE_PRIVATE_KEY') else None,
    "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
    "client_id": os.getenv("FIREBASE_CLIENT_ID"),
    "auth_uri": os.getenv("FIREBASE_AUTH_URI"),
    "token_uri": os.getenv("FIREBASE_TOKEN_URI"),
    "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_X509_CERT_URL"),
    "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_X509_CERT_URL"),
    "universe_domain": os.getenv("FIREBASE_UNIVERSE_DOMAIN")
}


class StorageService:
    def __init__(self):
        """Initialize storage service with data and model bucket configurations."""
        try:
            self.storage_client = storage.Client(
                credentials=service_account.Credentials.from_service_account_info(cert_dict)
            )
            self.data_bucket = self.storage_client.bucket(os.getenv("GOOGLE_BUCKET_DATA"))
            self.model_bucket = self.storage_client.bucket(os.getenv("GOOGLE_BUCKET_MODEL"))
            logger.info("Storage service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize storage service: {str(e)}")
            raise

    def save_data(self, content: dict, filename: str) -> bool:
        """Save data to the data bucket."""
        try:
            blob = self.data_bucket.blob(f"wikipedia/{filename}")
            blob.upload_from_string(
                json.dumps(content, indent=2, ensure_ascii=False),
                content_type='application/json'
            )
            logger.info(f"Data saved successfully to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            return False

    def load_data(self, filename: str) -> dict:
        """Load data from the data bucket."""
        try:
            blob = self.data_bucket.blob(f"wikipedia/{filename}")
            content = blob.download_as_string()
            return json.loads(content)
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None

    def save_model(self, model_data: bytes, filename: str) -> bool:
        """Save model to the model bucket."""
        try:
            blob = self.model_bucket.blob(filename)
            blob.upload_from_string(
                model_data,
                content_type='application/octet-stream'
            )
            logger.info(f"Model saved successfully to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False

    def load_model(self, filename: str) -> bytes:
        """Load model from the model bucket."""
        try:
            blob = self.model_bucket.blob(filename)
            return blob.download_as_bytes()
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None

    def list_data_files(self, prefix: str = "wikipedia/") -> list:
        """List all files in the data bucket with given prefix."""
        try:
            blobs = self.data_bucket.list_blobs(prefix=prefix)
            return [blob.name for blob in blobs]
        except Exception as e:
            logger.error(f"Error listing data files: {str(e)}")
            return []

    def list_models(self) -> list:
        """List all models in the model bucket."""
        try:
            blobs = self.model_bucket.list_blobs()
            return [blob.name for blob in blobs]
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return [] 
