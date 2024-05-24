import os
from datasets import load_dataset
from transformers import AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load your text file as a dataset
dataset = load_dataset('text', data_files={'train': 'combined_codebase.txt'})

# Choose a pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-base')
# Add a padding token if necessary
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Save the tokenized dataset
tokenized_dataset.save_to_disk('tokenized_codebase')

for index in range(3):  # Log the first 3 samples
    logger.info(f"Tokenized sample {index}: {tokenized_dataset['train'][index]}")
