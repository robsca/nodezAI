import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.utils.checkpoint as checkpoint

class SimpleTransformer(nn.Module):
    def __init__(self, num_tokens, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        # Store parameters as attributes
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_layers  # Using num_encoder_layers to match the attribute name used in get_model_status
        
        # Initialize layers
        self.embedding = nn.Embedding(num_tokens, d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead)
        self.transformer = TransformerEncoder(encoder_layers, num_layers)
        self.fc_out = nn.Linear(d_model, num_tokens)
        
        # Gradient checkpointing flag
        self.use_checkpoint = False
        
    def forward(self, x):
        # x shape: [batch_size, sequence_length]
        x = self.embedding(x)  # Shape: [batch_size, sequence_length, d_model]
        
        # Transformer expects shape [sequence_length, batch_size, d_model]
        x = x.transpose(0, 1)
        
        if self.use_checkpoint:
            # Use gradient checkpointing for transformer layers
            x = checkpoint.checkpoint(self.transformer, x)
        else:
            x = self.transformer(x)
            
        # Return to original shape [batch_size, sequence_length, d_model]
        x = x.transpose(0, 1)
        
        return self.fc_out(x)  # Shape: [batch_size, sequence_length, num_tokens]
        
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing."""
        self.use_checkpoint = True
        
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.use_checkpoint = False
