import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
import pandas as pd
import random
from typing import Tuple
import re

class TextToSQLDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question, schema, sql = self.data[idx]
        return self.process_data(question, schema, sql)
    
    def process_data(self, question, schema, sql):
        src = f"{question} {schema}"
        tgt = sql
        
        src_encoding = self.tokenizer(src, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        tgt_encoding = self.tokenizer(tgt, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        
        return {
            'src_input_ids': src_encoding['input_ids'].squeeze(),
            'src_attention_mask': src_encoding['attention_mask'].squeeze(),
            'tgt_input_ids': tgt_encoding['input_ids'].squeeze(),
            'tgt_attention_mask': tgt_encoding['attention_mask'].squeeze()
        }

    def apply_erosion(schema: str, sql: str,p_drop:int,p_add:int,schemas) -> Tuple[str, str]:
            sql_cols = re.findall(r'<col\d+>', sql)
            tables = re.findall(r'<table\d+>\[(.*?)\]\s\{(.*?)\}', schema)
            table_nums = re.findall(r'<col(\d+)>', schema)
            table_nums = list(map(int, table_nums))
            max_table_num = max(table_nums)+1
            modified_tables = []

            for table in tables:
                modified_table=[]
                table_name=table[0]
                columns = re.findall(r'<col\d+>\[.*?\]:\[.*?\]', table[1])
                
                # Permutation: Randomly reorder the columns
                random.shuffle(columns)
                
                # Removal: Randomly remove columns based on the drop probability
                for col in columns:
                    if random.random() < p_drop:
                        columns.remove(col)
                        removed_col = re.search(r'<col\d+>', col).group(0)
                        sql_cols = list(map(lambda x: x.replace(removed_col, "<unk>"), sql_cols))
                        sql = sql.replace(removed_col,"<unk>")
                        
                        
                    
                
                # Addition: Add columns from other schemas based on the add probability
                if random.random() < p_add:
                    extra_table = random.choice(schemas)
                    extra_columns = re.findall(r'\[.*?\]:\[.*?\]', extra_table)
                    
                    added_cols= random.sample(extra_columns, k=min(len(extra_columns), 1))
                    for col in added_cols:
                        modified_col = f'<col{max_table_num}>{col}'
                        max_table_num += 1
                        columns.append(modified_col)
                
                
                modified_table.append(table_name)
                modified_table.append(" ".join(columns))
                modified_tables.append(modified_table)
            
            modified_schema = " ".join([f"<table{i}>[{table[0]}] {{{table[1]}}}" for i, table in enumerate(modified_tables)])
            
            return modified_schema, sql

    def shuffle(sql: str) -> str:
        # Define regex patterns for <col>, <table>, and numbers
        col_pattern = r'<col\d+>'
        table_pattern = r'<table\d+>'
        number_pattern = r'\b\d+\b'  # Matches standalone numbers

        # Function to shuffle a list of items preserving non-matching items
        def shuffle_entities(entities, tokens):
            shuffled_entities = entities[:]
            random.shuffle(shuffled_entities)
            result = []
            index = 0
            for token in tokens:
                if token in entities:
                    result.append(shuffled_entities[index])
                    index += 1
                else:
                    result.append(token)
            return result

        # Tokenize SQL by spaces
        split_pattern = r'\s+|([`()"\'\[\]])'
        tokens = re.split(split_pattern,sql)
        tokens = filter(None, tokens)

        # Collect all <col>, <table>, and number entities
        entities_to_shuffle = []

        cols = re.findall(col_pattern,sql)
        tables = re.findall(table_pattern,sql)
        nums = re.findall(number_pattern,sql)

        entities_to_shuffle = cols + tables + nums
        
        # Shuffle entities while preserving the rest of the tokens
        shuffled_tokens = shuffle_entities(entities_to_shuffle, tokens)
        # Reconstruct SQL query
        shuffled_sql = " ".join(shuffled_tokens)
        return shuffled_sql


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0