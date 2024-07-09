from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path

import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter


def train_one_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    
    for batch in data_loader:
        src_input_ids = batch['src_input_ids'].to(device)
        src_attention_mask = batch['src_attention_mask'].to(device)
        tgt_input_ids = batch['tgt_input_ids'].to(device)
        tgt_attention_mask = batch['tgt_attention_mask'].to(device)
        
        optimizer.zero_grad()
        
        output = model(src_input_ids, tgt_input_ids)
        
        loss = criterion(output.view(-1, output.shape[-1]), tgt_input_ids.view(-1))
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(data_loader)

def evaluate(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in data_loader:
            src_input_ids = batch['src_input_ids'].to(device)
            src_attention_mask = batch['src_attention_mask'].to(device)
            tgt_input_ids = batch['tgt_input_ids'].to(device)
            tgt_attention_mask = batch['tgt_attention_mask'].to(device)
            
            output = model(src_input_ids, tgt_input_ids)
            
            loss = criterion(output.view(-1, output.shape[-1]), tgt_input_ids.view(-1))
            epoch_loss += loss.item()
    
    return epoch_loss / len(data_loader)

# Parameters
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
MAX_LEN = 128

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerModel(input_dim=768, output_dim=tokenizer.vocab_size, n_layers=3, heads=8, pf_dim=2048, dropout=0.1)
model = model.to(device)

# Optimizer and Criterion
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Training Loop
for epoch in range(EPOCHS):
    train_loss = train_one_epoch(model, data_loader, optimizer, criterion, device)
    eval_loss = evaluate(model, data_loader, criterion, device)
    print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")
