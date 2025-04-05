import torch
import torch.nn as nn
import time
import argparse
from datetime import datetime
from core.storage import StorageService
import io
import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def display_logo():
    """Display the NODEZ ASCII art logo"""
    logo = """
    \033[36m
    ███╗   ██╗ ██████╗ ██████╗ ███████╗███████╗
    ████╗  ██║██╔═══██╗██╔══██╗██╔════╝╚══███╔╝
    ██╔██╗ ██║██║   ██║██║  ██║█████╗    ███╔╝ 
    ██║╚██╗██║██║   ██║██║  ██║██╔══╝   ███╔╝  
    ██║ ╚████║╚██████╔╝██████╔╝███████╗███████╗
    ╚═╝  ╚═══╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝
    \033[0m
    \033[33m🤖 AI Model Management System\033[0m
    \033[90m----------------------------------------\033[0m
    """
    print(logo)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a GPT-2 model with custom configuration')
    parser.add_argument('--model-name', type=str, required=True,
                      help='Name for the model. Will be used in checkpoints and final model save.')
    parser.add_argument('--num-articles', type=int, default=1000,
                      help='Number of Wikipedia articles to download (default: 1000)')
    parser.add_argument('--min-freq', type=int, default=2,
                      help='Minimum frequency for vocabulary (default: 2)')
    parser.add_argument('--block-size', type=int, default=128,
                      help='Block size for training (default: 128)')
    parser.add_argument('--batch-size', type=int, default=4,
                      help='Batch size for training (default: 4)')
    return parser.parse_args()

from model import GPT2Config, GPT2, prepare_wikipedia_data

def get_available_models(storage_service):
    """Get list of available models from storage"""
    models = {}
    model_files = storage_service.list_models()
    
    # Group checkpoints by model name
    for file in model_files:
        if file.endswith('.pt'):
            if '_checkpoint_' in file:
                model_name = file.split('_checkpoint_')[0]
            elif '_final_model.pt' in file:
                model_name = file.split('_final_model.pt')[0]
            else:
                continue
                
            if model_name not in models:
                models[model_name] = {'checkpoints': [], 'final_model': None}
                
            if '_final_model.pt' in file:
                models[model_name]['final_model'] = file
            else:
                models[model_name]['checkpoints'].append(file)
    
    return models

def display_model_selection(models):
    """Display available models and return user selection"""
    while True:
        print("\nModels available:")
        model_names = list(models.keys())
        
        for idx, name in enumerate(model_names, 1):
            checkpoint_count = len(models[name]['checkpoints'])
            has_final = "Yes" if models[name]['final_model'] else "No"
            print(f"{idx}. {name} (Checkpoints: {checkpoint_count}, Final model: {has_final})")
        
        print(f"{len(model_names) + 1}. New Model")
        
        try:
            choice = input("\nSelect a model (enter number): ").strip()
            choice = int(choice)
            
            if 1 <= choice <= len(model_names):
                return model_names[choice - 1], True  # existing model
            elif choice == len(model_names) + 1:
                return get_model_name(), False  # new model
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def get_model_name():
    while True:
        print("\nPlease type the name of the model: ", end='')
        model_name = input().strip()
        
        if not model_name:
            print("Model name cannot be empty. Please try again.")
            continue
            
        print(f"\nThis will be the model name that will be used when saving in GCP: '{model_name}'")
        print("Are you sure? [Y/N]: ", end='')
        
        confirmation = input().strip().upper()
        if confirmation == 'Y':
            return model_name
        elif confirmation == 'N':
            continue
        else:
            print("Invalid input. Please enter Y or N.")
            continue

def load_model_checkpoint(storage_service, model_name, models):
    """Load the latest checkpoint or final model for the given model name"""
    model_files = models[model_name]
    
    if model_files['final_model']:
        checkpoint_path = model_files['final_model']
    elif model_files['checkpoints']:
        # Sort checkpoints by epoch and batch number
        checkpoints = sorted(model_files['checkpoints'], 
                           key=lambda x: [int(n) for n in x.replace('.pt','').split('_')[-2:] if n.isdigit()])
        checkpoint_path = checkpoints[-1]  # Get the latest checkpoint
    else:
        return None, None, None, 0, None
    
    # Load checkpoint from storage
    checkpoint_data = storage_service.load_model(checkpoint_path)
    if checkpoint_data:
        checkpoint = torch.load(io.BytesIO(checkpoint_data))
        # Get the vocabulary size from the model state dict
        vocab_size = checkpoint['model_state_dict']['tok_emb.weight'].shape[0]
        return (checkpoint['model_state_dict'], 
                checkpoint['optimizer_state_dict'], 
                checkpoint['loss'],
                checkpoint['epoch'],
                vocab_size)
    return None, None, None, 0, None

def confirm_training():
    while True:
        print("\nDo you want to start the training? [Y/N]: ", end='')
        choice = input().strip().upper()
        if choice in ['Y', 'N']:
            return choice == 'Y'
        print("Invalid input. Please enter Y or N.")

def select_operation():
    """Display operation options and get user selection"""
    while True:
        print("\n\033[33mSelect Operation:\033[0m")
        print("1. Training")
        print("2. Get Info")
        print("3. Test Model")
        print("4. Delete Model")
        
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                return int(choice)
            print("Invalid choice. Please enter 1, 2, 3, or 4.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_model_info(model_name, models, storage_service):
    """Display detailed information about the model"""
    print(f"\n\033[33mModel Information for {model_name}:\033[0m")
    print("\033[90m----------------------------------------\033[0m")
    
    model_data = models[model_name]
    
    # Display checkpoint information
    print(f"Number of checkpoints: {len(model_data['checkpoints'])}")
    if model_data['checkpoints']:
        print("\nCheckpoints:")
        for checkpoint in sorted(model_data['checkpoints']):
            print(f"  - {checkpoint}")
    
    # Display final model information
    if model_data['final_model']:
        print(f"\nFinal model: {model_data['final_model']}")
    else:
        print("\nNo final model available")
    
    # Try to load the latest checkpoint or final model to get more details
    if model_data['final_model'] or model_data['checkpoints']:
        checkpoint_path = model_data['final_model'] if model_data['final_model'] else sorted(model_data['checkpoints'])[-1]
        try:
            checkpoint_data = storage_service.load_model(checkpoint_path)
            if checkpoint_data:
                checkpoint = torch.load(io.BytesIO(checkpoint_data))
                print(f"\nLast training epoch: {checkpoint['epoch']}")
                print(f"Last loss value: {checkpoint['loss']:.4f}")
        except Exception as e:
            print(f"\nError loading model details: {str(e)}")
    
    print("\033[90m----------------------------------------\033[0m")

def delete_model(model_name, models, storage_service):
    """Delete a model and all its checkpoints from storage"""
    print(f"\n\033[33mDeleting model: {model_name}\033[0m")
    print("\033[31mWARNING: This action cannot be undone!\033[0m")
    print("Are you sure? [Y/N]: ", end='')
    
    confirmation = input().strip().upper()
    if confirmation != 'Y':
        print("Deletion cancelled.")
        return
    
    model_data = models[model_name]
    deleted_files = 0
    
    # Delete checkpoints
    for checkpoint in model_data['checkpoints']:
        if storage_service.delete_model(checkpoint):
            deleted_files += 1
            print(f"Deleted checkpoint: {checkpoint}")
    
    # Delete final model if it exists
    if model_data['final_model']:
        if storage_service.delete_model(model_data['final_model']):
            deleted_files += 1
            print(f"Deleted final model: {model_data['final_model']}")
    
    print(f"\nSuccessfully deleted {deleted_files} files for model {model_name}")

def generate_text(model, prompt, max_length=100, temperature=0.7):
    """Generate text from the model given a prompt"""
    model.eval()
    with torch.no_grad():
        # Convert prompt to tensor
        prompt_tensor = torch.tensor([ord(c) for c in prompt], dtype=torch.long).unsqueeze(0)
        prompt_tensor = prompt_tensor.to(model.device)
        
        # Generate text
        generated = list(prompt)
        for _ in range(max_length):
            # Get model predictions
            outputs = model(prompt_tensor)
            next_token_logits = outputs[0, -1, :] / temperature
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
            
            # Convert to character and append
            generated.append(chr(next_token.item()))
            
            # Update input tensor
            prompt_tensor = torch.cat([prompt_tensor, next_token.unsqueeze(0)], dim=1)
            
            # Stop if we generate a newline
            if generated[-1] == '\n':
                break
                
        return ''.join(generated)

def chat_mode(model, model_name):
    """Interactive chat mode with the model"""
    print(f"\n\033[33mChat Mode - {model_name}\033[0m")
    print("\033[90m----------------------------------------\033[0m")
    print("Type 'exit' to quit chat mode")
    print("Type 'clear' to clear the conversation")
    print("\033[90m----------------------------------------\033[0m\n")
    
    while True:
        try:
            prompt = input("\033[32mYou:\033[0m ").strip()
            
            if prompt.lower() == 'exit':
                break
            elif prompt.lower() == 'clear':
                os.system('clear' if os.name == 'posix' else 'cls')
                continue
            elif not prompt:
                continue
                
            print("\n\033[34mNODEZ:\033[0m", end=' ')
            response = generate_text(model, prompt)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\nExiting chat mode...")
            break
        except Exception as e:
            print(f"\n\033[31mError: {str(e)}\033[0m")

if __name__ == '__main__':
    try:
        # Display the NODEZ logo
        display_logo()
        
        # Initialize storage service
        storage_service = StorageService()
        
        # Get available models and let user select one
        models = get_available_models(storage_service)
        model_name, is_existing_model = display_model_selection(models)
        
        # Get operation selection
        operation = select_operation()
        
        if operation == 1:  # Training
            # Confirm training
            if not confirm_training():
                print("Training cancelled by user.")
                sys.exit(0)
            
            start_time = time.time()
            logger.info(f'Training started at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            logger.info(f'Model name: {model_name}')
            
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
            
            # Initialize optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Load checkpoint if it's an existing model
            if is_existing_model:
                model_state, optimizer_state, last_loss, start_epoch, vocab_size = load_model_checkpoint(
                    storage_service, model_name, models
                )
                
                if model_state is not None:
                    model.load_state_dict(model_state)
                    optimizer.load_state_dict(optimizer_state)
                    logger.info(f'Loaded checkpoint for model {model_name}. Starting from epoch {start_epoch + 1}')
                else:
                    logger.warning(f'No checkpoint found for model {model_name}. Starting from scratch.')
                    start_epoch = 0
            else:
                start_epoch = 0
            
            criterion = nn.CrossEntropyLoss()
            
            # Train the model
            train(model, train_loader, optimizer, criterion, num_epochs=1, storage_service=storage_service, model_name=model_name)
            
            end_time = time.time()
            total_time = end_time - start_time
            logger.info(f'Training completed at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            logger.info(f'Total execution time: {total_time:.2f}s')
            
        elif operation == 2:  # Get Info
            get_model_info(model_name, models, storage_service)
            
        elif operation == 3:  # Test Model
            # Load the model checkpoint first to get the vocabulary size
            model_state, _, _, _, vocab_size = load_model_checkpoint(storage_service, model_name, models)
            
            if model_state is not None and vocab_size is not None:
                # Initialize model with the correct vocabulary size
                config = GPT2Config(
                    vocab_size=vocab_size,
                    n_positions=1024,
                    n_embd=768,
                    n_layer=12,
                    n_head=12,
                    dropout=0.1
                )
                model = GPT2(config)
                
                # Load the state dict
                model.load_state_dict(model_state)
                print(f"\n\033[32mModel {model_name} loaded successfully!\033[0m")
                print(f"Vocabulary size: {vocab_size}")
                chat_mode(model, model_name)
            else:
                print(f"\n\033[31mError: Could not load model {model_name}\033[0m")
                
        elif operation == 4:  # Delete Model
            delete_model(model_name, models, storage_service)
        
    except KeyboardInterrupt:
        print("\nOperation interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)
