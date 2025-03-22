from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
from pathlib import Path
import asyncio
import json
from core.train import train_model
from core.prepare_data import DataPreparator
from core.download_data import WikipediaDownloader
from core.storage import StorageService
from core.model_base import SimpleTransformer
import logging
from dotenv import load_dotenv
import io
import os
import pickle
from core.models.gpt2 import GPT2, GPT2Config

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# Mount templates and static files
templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize WikipediaDownloader and StorageService
downloader = WikipediaDownloader(output_dir="data/wikipedia")
storage_service = StorageService()

# Check if we're in test mode
is_test = os.getenv('TEST', 'false').lower() == 'true'

# Load model and data preparator
model = None
data_preparator = None

def load_model_and_preparator():
    global model, data_preparator
    try:
        if is_test:
            # Load from local file
            model_path = Path('core/models/model.pth')  # Default model path
            if model_path.exists():
                checkpoint = torch.load(model_path, pickle_module=pickle)
                data_preparator = checkpoint['data_preparator']
                
                # Get model parameters from checkpoint
                params = checkpoint['model_params']
                model_type = params.get('model_type', 'simple')  # Default to simple for backward compatibility
                
                # Initialize model based on type
                if model_type == 'gpt2':
                    config = GPT2Config(
                        vocab_size=params['vocab_size'],
                        n_positions=params['n_positions'],
                        n_embd=params['n_embd'],
                        n_layer=params['n_layer'],
                        n_head=params['n_head'],
                        dropout=params['dropout']
                    )
                    model = GPT2(config)
                else:
                    model = SimpleTransformer(
                        num_tokens=params['vocab_size'],
                        d_model=params['n_embd'],
                        nhead=params['n_head'],
                        num_layers=params['n_layer']
                    )
                
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                logger.info(f"Model loaded from local file (type: {model_type})")
        else:
            # Load from storage bucket
            model_bytes = storage_service.load_model('model.pth')  # Default model name
            if model_bytes:
                buffer = io.BytesIO(model_bytes)
                checkpoint = torch.load(buffer, pickle_module=pickle)
                data_preparator = checkpoint['data_preparator']
                
                # Get model parameters from checkpoint
                params = checkpoint['model_params']
                model_type = params.get('model_type', 'simple')  # Default to simple for backward compatibility
                
                # Initialize model based on type
                if model_type == 'gpt2':
                    config = GPT2Config(
                        vocab_size=params['vocab_size'],
                        n_positions=params['n_positions'],
                        n_embd=params['n_embd'],
                        n_layer=params['n_layer'],
                        n_head=params['n_head'],
                        dropout=params['dropout']
                    )
                    model = GPT2(config)
                else:
                    model = SimpleTransformer(
                        num_tokens=params['vocab_size'],
                        d_model=params['n_embd'],
                        nhead=params['n_head'],
                        num_layers=params['n_layer']
                    )
                
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                logger.info(f"Model loaded from storage bucket (type: {model_type})")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        model = None
        data_preparator = None

# Load model on startup
load_model_and_preparator()

class TrainInput(BaseModel):
    model_name: str
    model_type: str = 'simple'
    vocab_size: int
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    dropout: float = 0.1
    batch_size: int = 32
    num_epochs: int = 10

class PredictInput(BaseModel):
    input_text: str
    max_length: int = 50
    temperature: float = 1.0

class DownloadInput(BaseModel):
    num_articles: int
    max_workers: int = 4

class LoadModelInput(BaseModel):
    model_name: str

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("base.html", {"request": request})

@app.get("/train")
async def train_page(request: Request):
    return templates.TemplateResponse("train.html", {"request": request})

@app.get("/test")
async def test_page(request: Request):
    return templates.TemplateResponse("test.html", {"request": request})

@app.get("/download")
async def download_page(request: Request):
    return templates.TemplateResponse("download.html", {"request": request})

@app.post("/api/train")
async def train(data: TrainInput):
    try:
        logger.info(f"Starting training with parameters: {data.dict()}")
        
        # Validate model type
        if data.model_type not in ['simple', 'gpt2']:
            logger.error(f"Invalid model type: {data.model_type}")
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": f"Invalid model type: {data.model_type}. Must be 'simple' or 'gpt2'."}
            )
        
        # Validate model parameters
        if data.vocab_size < 100:
            logger.error(f"Invalid vocabulary size: {data.vocab_size}")
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Vocabulary size must be at least 100"}
            )
        
        if data.n_positions < 64:
            logger.error(f"Invalid sequence length: {data.n_positions}")
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Maximum sequence length must be at least 64"}
            )
        
        if data.n_embd < 64:
            logger.error(f"Invalid embedding dimension: {data.n_embd}")
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Embedding dimension must be at least 64"}
            )
        
        if data.n_head < 1:
            logger.error(f"Invalid number of attention heads: {data.n_head}")
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Number of attention heads must be at least 1"}
            )
        
        if data.n_layer < 1:
            logger.error(f"Invalid number of layers: {data.n_layer}")
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Number of layers must be at least 1"}
            )
        
        if not (0 <= data.dropout <= 1):
            logger.error(f"Invalid dropout rate: {data.dropout}")
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Dropout rate must be between 0 and 1"}
            )
        
        # Get the data path based on test mode
        if is_test:
            data_path = "wikipedia_articles.json"  # Just the filename for local
            # Check if the file exists
            if not Path("data/wikipedia").exists() or not Path("data/wikipedia/wikipedia_articles.json").exists():
                logger.error("Training data not found in test mode")
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": "No training data found. Please download articles first."}
                )
            logger.info("Using local training data file")
        else:
            # Get the most recent file from storage
            files = storage_service.list_data_files()
            if not files:
                logger.error("No training data files found in storage")
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": "No training data available. Please download articles first."}
                )
            # Get the most recent file and remove the 'wikipedia/' prefix
            data_path = files[-1].replace('wikipedia/', '')
            logger.info(f"Using storage training data file: {data_path}")
        
        # Create models directory if it doesn't exist
        if is_test:
            model_dir = Path('core/models')
            model_dir.mkdir(exist_ok=True, parents=True)
            logger.info("Created models directory")
        
        # Start training
        try:
            logger.info("Starting model training...")
            await train_model(
                data_path=data_path,
                model_name=data.model_name,
                model_type=data.model_type,
                vocab_size=data.vocab_size,
                n_positions=data.n_positions,
                n_embd=data.n_embd,
                n_layer=data.n_layer,
                n_head=data.n_head,
                dropout=data.dropout,
                batch_size=data.batch_size,
                num_epochs=data.num_epochs
            )
            logger.info("Model training completed successfully")
        except Exception as train_error:
            logger.error(f"Training error: {str(train_error)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": f"Training failed: {str(train_error)}"}
            )
        
        # Reload the model
        try:
            logger.info("Reloading trained model...")
            load_model_and_preparator()
            logger.info("Model reloaded successfully")
        except Exception as load_error:
            logger.error(f"Error loading trained model: {str(load_error)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": f"Model trained but failed to load: {str(load_error)}"}
            )
        
        return {"status": "success", "message": f"Model trained and saved as {data.model_name}"}
    except Exception as e:
        logger.error(f"Unexpected error during training: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Unexpected error: {str(e)}"}
        )

@app.post("/api/predict")
async def predict(data: PredictInput):
    if model is None or data_preparator is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Model not trained yet. Please train the model first."}
        )
    
    try:
        # Convert input text to sequence
        input_sequence = data_preparator.text_to_sequence(data.input_text)
        # Shape: [batch_size=1, sequence_length]
        input_tensor = torch.tensor(input_sequence).unsqueeze(0)
        
        # Generate text
        with torch.no_grad():
            output_sequence = []
            current_input = input_tensor
            
            # Keep track of consecutive UNK tokens
            consecutive_unks = 0
            max_consecutive_unks = 3  # Stop after this many consecutive UNKs
            
            for _ in range(data.max_length):
                # Forward pass through the model
                # output shape: [batch_size=1, sequence_length, vocab_size]
                output = model(current_input)
                
                # Get next token probabilities from the last position
                # Shape: [batch_size=1, vocab_size]
                next_token_logits = output[0, -1, :].unsqueeze(0)  # Keep batch dimension
                
                # Apply temperature and softmax
                scaled_logits = next_token_logits / (data.temperature if data.temperature > 0 else 1.0)
                probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
                
                # Get top-k probabilities and indices
                top_k = 40
                top_k_probs, top_k_indices = torch.topk(probs, k=min(top_k, probs.size(-1)))
                
                # Sample from top-k
                selected_idx = torch.multinomial(top_k_probs, num_samples=1)
                next_token = top_k_indices[0][selected_idx[0]]
                
                # Check for UNK token
                if next_token.item() == data_preparator.word_to_idx.get('<UNK>', 0):
                    consecutive_unks += 1
                    if consecutive_unks >= max_consecutive_unks:
                        break
                else:
                    consecutive_unks = 0
                
                # Stop if we predict the end token
                if next_token.item() == data_preparator.word_to_idx.get('<PAD>', 0):
                    break
                
                output_sequence.append(next_token.item())
                
                # Prepare next input by concatenating
                # Shape: [batch_size=1, 1]
                next_token = next_token.view(1, 1)
                current_input = torch.cat([current_input, next_token], dim=1)
                
                # Break if the sequence is getting too long
                if len(output_sequence) >= data.max_length:
                    break
        
        # Convert output sequence to text
        generated_text = data_preparator.sequence_to_text(output_sequence)
        
        # Remove repeated spaces and clean up the text
        generated_text = ' '.join(generated_text.split())
        
        # If the generated text is empty or only contains special tokens, return a message
        if not generated_text or all(token in ['<UNK>', '<PAD>'] for token in generated_text.split()):
            return {"generated_text": data.input_text + " [Unable to generate meaningful continuation]"}
        
        # Combine input and generated text
        full_text = data.input_text + " " + generated_text
        
        return {"generated_text": full_text}
    except Exception as e:
        logger.error(f"Error during text generation: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/api/download")
async def download_articles(data: DownloadInput):
    try:
        # Create a background task for downloading
        def download_task():
            downloader.max_workers = data.max_workers
            articles = downloader.get_random_articles(data.num_articles)
            downloader._save_articles(articles, "wikipedia_articles.json")
            
            # Calculate statistics
            total_words = sum(len(article['text'].split()) for article in articles)
            avg_words = total_words / len(articles)
            
            # Calculate category distribution
            category_counts = {}
            for article in articles:
                category = article['category']
                category_counts[category] = category_counts.get(category, 0) + 1
            
            category_distribution = {}
            for category, count in category_counts.items():
                percentage = (count / len(articles)) * 100
                category_distribution[category] = {
                    "count": count,
                    "percentage": percentage
                }
            
            return {
                "total_articles": len(articles),
                "total_words": total_words,
                "avg_words": avg_words,
                "category_distribution": category_distribution
            }
        
        # Run the download task in a thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, download_task)
        
        return result
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/api/data/stats")
async def get_data_stats():
    try:
        if not downloader.is_test:
            # Get list of files from storage
            files = downloader.storage_service.list_data_files()
            stats = []
            
            for file in files:
                try:
                    data = downloader.storage_service.load_data(file.replace('wikipedia/', ''))
                    if data:
                        # Calculate statistics
                        total_words = sum(len(article['text'].split()) for article in data)
                        avg_words = total_words / len(data)
                        
                        # Calculate category distribution
                        category_counts = {}
                        for article in data:
                            category = article['category']
                            category_counts[category] = category_counts.get(category, 0) + 1
                        
                        category_distribution = {}
                        for category, count in category_counts.items():
                            percentage = (count / len(data)) * 100
                            category_distribution[category] = {
                                "count": count,
                                "percentage": percentage
                            }
                        
                        stats.append({
                            "filename": file,
                            "total_articles": len(data),
                            "total_words": total_words,
                            "avg_words": avg_words,
                            "category_distribution": category_distribution
                        })
                except Exception as e:
                    logger.error(f"Error processing file {file}: {str(e)}")
                    continue
            
            return {"files": stats}
        else:
            # For local testing, read from the data directory
            data_dir = Path("data/wikipedia")
            stats = []
            
            if data_dir.exists():
                for file in data_dir.glob("*.json"):
                    try:
                        with open(file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                        total_words = sum(len(article['text'].split()) for article in data)
                        avg_words = total_words / len(data)
                        
                        category_counts = {}
                        for article in data:
                            category = article['category']
                            category_counts[category] = category_counts.get(category, 0) + 1
                        
                        category_distribution = {}
                        for category, count in category_counts.items():
                            percentage = (count / len(data)) * 100
                            category_distribution[category] = {
                                "count": count,
                                "percentage": percentage
                            }
                        
                        stats.append({
                            "filename": file.name,
                            "total_articles": len(data),
                            "total_words": total_words,
                            "avg_words": avg_words,
                            "category_distribution": category_distribution
                        })
                    except Exception as e:
                        logger.error(f"Error processing file {file}: {str(e)}")
                        continue
            
            return {"files": stats}
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/api/model/status")
async def get_model_status():
    try:
        model_loaded = model is not None and data_preparator is not None
        
        if model_loaded:
            # Get model parameters directly from model attributes
            parameters = {
                "num_tokens": model.num_tokens,
                "d_model": model.d_model,
                "nhead": model.nhead,
                "num_layers": model.num_encoder_layers
            }
            
            # Get model file info
            if is_test:
                model_path = Path('core/models/model.pth')
                model_file = "model.pth"
                last_modified = model_path.stat().st_mtime * 1000  # Convert to milliseconds
            else:
                model_file = "model.pth"
                # For cloud storage, we don't have direct access to last modified time
                last_modified = None
            
            return {
                "model_loaded": True,
                "parameters": parameters,
                "is_test": is_test,
                "model_file": model_file,
                "last_modified": last_modified
            }
        else:
            return {
                "model_loaded": False
            }
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/api/models")
async def list_models():
    try:
        models = []
        if is_test:
            # List models from local directory
            model_dir = Path('core/models')
            if model_dir.exists():
                models = [f.name for f in model_dir.glob('*.pth')]
        else:
            # List models from storage bucket
            models = storage_service.list_models()
        
        return {"models": models}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/api/load_model")
async def load_specific_model(data: LoadModelInput):
    try:
        global model, data_preparator
        
        if is_test:
            # Load from local file
            model_path = Path('core/models') / data.model_name
            if not model_path.exists():
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Model {data.model_name} not found"}
                )
            
            checkpoint = torch.load(model_path, pickle_module=pickle)
            data_preparator = checkpoint['data_preparator']
            
            # Get model parameters from checkpoint
            params = checkpoint['model_params']
            
            # Initialize model and load state dict
            model = SimpleTransformer(
                num_tokens=params['vocab_size'],
                d_model=params['d_model'],
                nhead=params['nhead'],
                num_layers=params['num_layers']
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            logger.info(f"Model {data.model_name} loaded from local file")
        else:
            # Load from storage bucket
            model_bytes = storage_service.load_model(data.model_name)
            if not model_bytes:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Model {data.model_name} not found"}
                )
            
            buffer = io.BytesIO(model_bytes)
            checkpoint = torch.load(buffer, pickle_module=pickle)
            data_preparator = checkpoint['data_preparator']
            
            # Get model parameters from checkpoint
            params = checkpoint['model_params']
            
            # Initialize model and load state dict
            model = SimpleTransformer(
                num_tokens=params['vocab_size'],
                d_model=params['d_model'],
                nhead=params['nhead'],
                num_layers=params['num_layers']
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            logger.info(f"Model {data.model_name} loaded from storage bucket")
        
        return {"status": "success", "message": f"Model {data.model_name} loaded successfully"}
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8080)
