import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from datetime import datetime
import logging
import os
import json
from torch.utils.data import Dataset, DataLoader
from core.download_data import WikipediaDownloader
from core.prepare_data import DataPreparator, prepare_data
from core.storage import StorageService
import io
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WikipediaDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

class GPT2Config:
    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        dropout=0.1
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout

class AttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        
        self.q = nn.Linear(config.n_embd, self.head_dim)
        self.k = nn.Linear(config.n_embd, self.head_dim)
        self.v = nn.Linear(config.n_embd, self.head_dim)
        
    def forward(self, x, mask=None):
        q = self.q(x)  # (batch, seq_len, head_dim)
        k = self.k(x)  # (batch, seq_len, head_dim)
        v = self.v(x)  # (batch, seq_len, head_dim)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, mask=None):
        out = torch.cat([h(x, mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )
        
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ff = FeedForward(config)
        
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x

class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.n_positions, config.n_embd))
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        # Output head
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.apply(self._init_weights)
        
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, idx, mask=None):
        b, t = idx.size()
        assert t <= self.config.n_positions, f"Sequence length {t} exceeds maximum length {self.config.n_positions}"
        
        # Get token embeddings
        tok_emb = self.tok_emb(idx)  # (batch, seq_len, n_embd)
        
        # Add positional embeddings
        pos_emb = self.pos_emb[:, :t, :]  # (1, seq_len, n_embd)
        x = self.drop(tok_emb + pos_emb)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
            
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits

def train_step(model, optimizer, data, targets, criterion):
    start_time = time.time()
    model.train()
    optimizer.zero_grad()
    
    # Move data to GPU
    data = data.to(model.device)
    targets = targets.to(model.device)
    
    # Forward pass
    logits = model(data)
    # Reshape logits and targets for loss calculation
    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    step_time = time.time() - start_time
    return loss.item(), step_time

def save_checkpoint(model, optimizer, epoch, loss, storage_service, checkpoint_name):
    """Save model checkpoint to cloud storage"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    # Save checkpoint to bytes buffer
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)
    
    # Upload to cloud storage
    success = storage_service.save_model(buffer.getvalue(), checkpoint_name)
    if success:
        logger.info(f"Checkpoint saved successfully: {checkpoint_name}")
    else:
        logger.error(f"Failed to save checkpoint: {checkpoint_name}")
    
    return success

def train(model, train_loader, optimizer, criterion, num_epochs, storage_service):
    start_time = time.time()
    
    # Log device information
    if model.device.type == 'cuda':
        logger.info(f"Training on GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        logger.info(f"GPU Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    else:
        logger.info("Training on CPU")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        total_loss = 0
        total_time = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            loss, step_time = train_step(model, optimizer, data, targets, criterion)
            total_loss += loss
            total_time += step_time
            
            if batch_idx % 100 == 0:
                logger.info(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss:.4f}, Time/batch: {step_time:.3f}s')
                if model.device.type == 'cuda':
                    logger.info(f'GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB')
                
                # Save checkpoint every 100 batches
                checkpoint_name = f'gpt2_checkpoint_e{epoch+1}_b{batch_idx}.pt'
                save_checkpoint(model, optimizer, epoch, loss, storage_service, checkpoint_name)
        
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        logger.info(f'Epoch: {epoch+1}, Average Loss: {avg_loss:.4f}, Epoch Time: {epoch_time:.2f}s')
        
        # Save checkpoint at end of each epoch
        checkpoint_name = f'gpt2_checkpoint_epoch{epoch+1}.pt'
        save_checkpoint(model, optimizer, epoch, avg_loss, storage_service, checkpoint_name)
    
    total_time = time.time() - start_time
    logger.info(f'Total training time: {total_time:.2f}s')
    
    # Save final model
    final_model_name = 'gpt2_final_model.pt'
    save_checkpoint(model, optimizer, num_epochs, avg_loss, storage_service, final_model_name)

def prepare_wikipedia_data(output_dir="data/wikipedia", num_articles=1000, min_freq=2, block_size=128, batch_size=4):
    """Download and prepare Wikipedia data for training"""
    
    # Initialize storage service to check for existing data
    storage_service = StorageService()
    data_files = storage_service.list_data_files()
    
    if data_files:
        logger.info("Found existing data files, loading from storage...")
        try:
            # Load the first available data file
            articles = storage_service.load_data(data_files[0].split('/')[-1])
            texts = [article['text'] for article in articles]
            logger.info(f"Successfully loaded {len(texts)} articles from storage")
        except Exception as e:
            logger.error(f"Error loading existing data: {str(e)}")
            raise
    else:
        # Initialize downloader and preparator
        downloader = WikipediaDownloader(output_dir=output_dir)
        
        # Download Wikipedia articles
        logger.info("No existing data found. Downloading Wikipedia articles...")
        try:
            articles = downloader.get_random_articles(num_articles)
            logger.info(f"Successfully downloaded {len(articles)} articles")
            texts = [article['text'] for article in articles]
        except Exception as e:
            logger.error(f"Error downloading articles: {str(e)}")
            raise
    
    # Prepare data using the prepare_data function
    logger.info("Preparing data...")
    try:
        dataset, data_preparator = prepare_data(
            texts=texts,
            vocab_size=1000,
            seq_length=block_size,
            min_freq=min_freq
        )
        logger.info(f"Successfully prepared data with vocabulary size {len(data_preparator.vocab)}")
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader, data_preparator

if __name__ == '__main__':
    start_time = time.time()
    logger.info(f'Training started at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    
    # Initialize storage service
    storage_service = StorageService()
    
    # Prepare Wikipedia data
    train_loader, data_preparator = prepare_wikipedia_data(
        num_articles=1000,
        min_freq=2,
        block_size=128,
        batch_size=4
    )
    
    # Initialize model with vocabulary size from data_preparator
    config = GPT2Config(
        vocab_size=len(data_preparator.vocab),
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        dropout=0.1
    )
    model = GPT2(config)
    
    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train the model
    train(model, train_loader, optimizer, criterion, num_epochs=1, storage_service=storage_service)
    
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f'Training completed at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info(f'Total execution time: {total_time:.2f}s')
