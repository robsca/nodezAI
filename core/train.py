import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from core.model_base import SimpleTransformer
from core.models.gpt2 import GPT2, GPT2Config
from core.prepare_data import prepare_data
from core.storage import StorageService
import numpy as np
import os
from dotenv import load_dotenv
import io
import json
from pathlib import Path
import asyncio
import pickle

# Load environment variables
load_dotenv()

# Set PyTorch memory management settings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

async def train_model(
    data_path: str,
    model_name: str,
    model_type: str = 'simple',
    file_type: str = 'json',
    vocab_size=1000,
    n_positions=1024,
    n_embd=768,
    n_layer=12,
    n_head=12,
    dropout=0.1,
    batch_size=8,  # Reduced batch size
    num_epochs=10,
    learning_rate=0.001,
    max_samples=None,
    min_freq=2,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    # Ensure model name ends with .pth
    if not model_name.endswith('.pth'):
        model_name += '.pth'
    
    # Initialize model based on type
    if model_type == 'gpt2':
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            dropout=dropout
        )
        model = GPT2(config).to(device)
        # No gradient checkpointing for GPT2
    else:  # simple transformer
        model = SimpleTransformer(
            num_tokens=vocab_size,
            d_model=n_embd,
            nhead=n_head,
            num_layers=n_layer
        ).to(device)
        # Enable gradient checkpointing for SimpleTransformer
        model.gradient_checkpointing_enable()
    
    # Load data based on TEST environment variable
    is_test = os.getenv('TEST', 'false').lower() == 'true'
    print("Loading and preparing data...")
    
    try:
        # Run the CPU-intensive data loading and training in a thread pool
        loop = asyncio.get_event_loop()
        
        async def load_and_train():
            if is_test:
                # Load from local file
                data_file = Path("data/wikipedia") / data_path
                if not data_file.exists():
                    raise FileNotFoundError(f"Data file not found: {data_file}")
                
                with open(data_file, 'r', encoding='utf-8') as f:
                    articles = json.load(f)
            else:
                # Load from storage bucket
                storage_service = StorageService()
                articles = storage_service.load_data(data_path)
                if articles is None:
                    raise FileNotFoundError(f"Failed to load data from storage bucket: {data_path}")
            
            # Extract text from articles
            texts = [article['text'] for article in articles]
            
            # Prepare data with appropriate sequence length
            dataset, data_preparator = await loop.run_in_executor(None, lambda: prepare_data(
                texts=texts,
                vocab_size=vocab_size,
                seq_length=n_positions,
                max_samples=max_samples,
                min_freq=min_freq
            ))
            
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Training loop
            model.train()
            for epoch in range(num_epochs):
                total_loss = 0
                for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
                    # Allow other async operations to run
                    if batch_idx % 10 == 0:
                        await asyncio.sleep(0)
                        
                    input_seq = input_seq.to(device)
                    target_seq = target_seq.to(device)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass (handle both model types)
                    if model_type == 'gpt2':
                        output = model(input_seq)  # GPT2 expects indices directly
                    else:
                        output = model(input_seq)  # SimpleTransformer output
                    
                    # Reshape output and target for loss calculation
                    output = output.view(-1, vocab_size)
                    target = target_seq.view(-1)
                    
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                    # Clear CUDA cache after each batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    if batch_idx % 100 == 0:
                        print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
                
                avg_loss = total_loss / len(dataloader)
                print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
            
            # Save the trained model and data preparator
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'data_preparator': data_preparator,
                'model_params': {
                    'model_type': model_type,
                    'vocab_size': vocab_size,
                    'n_positions': n_positions,
                    'n_embd': n_embd,
                    'n_layer': n_layer,
                    'n_head': n_head,
                    'dropout': dropout
                }
            }
            
            if is_test:
                # Save locally
                model_dir = Path('core/models')
                model_dir.mkdir(exist_ok=True)
                model_path = model_dir / model_name
                torch.save(checkpoint, model_path, pickle_module=pickle)
                print(f"Model and data preparator saved locally as {model_name}!")
            else:
                # Save to Google Cloud Storage
                try:
                    # Convert model to bytes
                    buffer = io.BytesIO()
                    torch.save(checkpoint, buffer, pickle_module=pickle)
                    model_bytes = buffer.getvalue()
                    
                    # Save to storage bucket
                    success = storage_service.save_model(model_bytes, model_name)
                    if success:
                        print(f"Model and data preparator saved to storage bucket as {model_name}!")
                    else:
                        print("Failed to save model to storage bucket")
                        # Fallback to local save
                        model_dir = Path('core/models')
                        model_dir.mkdir(exist_ok=True)
                        model_path = model_dir / model_name
                        torch.save(checkpoint, model_path, pickle_module=pickle)
                        print(f"Model saved locally as {model_name} (fallback)")
                except Exception as e:
                    print(f"Error saving to storage bucket: {str(e)}")
                    # Fallback to local save
                    model_dir = Path('core/models')
                    model_dir.mkdir(exist_ok=True)
                    model_path = model_dir / model_name
                    torch.save(checkpoint, model_path, pickle_module=pickle)
                    print(f"Model saved locally as {model_name} (fallback)")
        
        await load_and_train()
                    
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a transformer model on text data')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data file')
    parser.add_argument('--model_name', type=str, required=True, help='Name to save the model with (e.g., my_model_v1)')
    parser.add_argument('--model_type', type=str, default='simple', choices=['simple', 'gpt2'], help='Type of transformer model')
    parser.add_argument('--file_type', type=str, default='json', choices=['txt', 'json', 'csv'], help='Type of input file')
    parser.add_argument('--vocab_size', type=int, default=1000, help='Vocabulary size')
    parser.add_argument('--n_positions', type=int, default=1024, help='Maximum sequence length')
    parser.add_argument('--n_embd', type=int, default=768, help='Embedding dimension')
    parser.add_argument('--n_layer', type=int, default=12, help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to use')
    parser.add_argument('--min_freq', type=int, default=2, help='Minimum word frequency for vocabulary')
    
    args = parser.parse_args()
    
    # Run the async function
    asyncio.run(train_model(
        data_path=args.data_path,
        model_name=args.model_name,
        model_type=args.model_type,
        file_type=args.file_type,
        vocab_size=args.vocab_size,
        n_positions=args.n_positions,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        dropout=args.dropout,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        max_samples=args.max_samples,
        min_freq=args.min_freq
    ))
