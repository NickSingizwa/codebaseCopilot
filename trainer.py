import os
from datasets import load_from_disk
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer

# Load the tokenized dataset
tokenized_dataset = load_from_disk('tokenized_codebase')

# Load a pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2-large')
tokenizer = AutoTokenizer.from_pretrained('gpt2-large')

# Add a padding token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

# Tokenize the dataset if not already tokenized
if 'train' not in tokenized_dataset:
    tokenized_dataset = tokenized_dataset.map(tokenize_function, batched=True)

# Data collator for language modeling (object responsible for preparing and batching training data before feeding it into our model.
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',              
    overwrite_output_dir=True,           
    num_train_epochs=1,                  # number of times the entire training dataset is passed through the model during training.
    per_device_train_batch_size=2,       # training examples that are processed together in a single iteration.
    save_steps=20000,                   # save checkpoint every 20,000 steps
    save_total_limit=2,                  # limit the total amount of checkpoints
    prediction_loss_only=True,           # only calculate the prediction loss
    logging_dir='./logs',            
    logging_steps=5, 
)

# Initialize Trainer
trainer = Trainer(
    model=model,                         
    args=training_args,                  
    data_collator=data_collator,         
    train_dataset=tokenized_dataset['train'], 
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')
