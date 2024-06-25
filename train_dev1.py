import os
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import json
import wandb
import tqdm
import re

from galore_torch import GaLoreAdamW

from src import Griffin
from model_config import GriffinConfig

# Initialize wandb
wandb.init(project="griffin", config={
    "num_batches": 1000,
    "batch_size": 4,
    "learning_rate": 0.0001,
    "gradient_accumulate_every": 4,
    "validate_every": 40,
    "vocab_size": 10000,
})

device = torch.device("cuda:1")

# Define paths
dataset_path = "processed_dataset.json"
tokenizer_path = "bpe_tokenizer-wikitext.json"
checkpoint_dir = "checkpoints_griffin"
os.makedirs(checkpoint_dir, exist_ok=True)

# Check if processed dataset exists
if not os.path.exists(dataset_path):
    # Load the dataset
    train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    val_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    
    # Function to remove non-Latin characters
    def remove_non_latin(text):
        return re.sub(r'[^a-zA-Z0-9\s.,!?\'"()\-:;]', '', text)
    
    # Convert text to lowercase
    def preprocess_text(dataset):
        return [remove_non_latin(text.lower()) for text in dataset["text"]]

    train_texts = preprocess_text(train_dataset)
    val_texts = preprocess_text(val_dataset)

    # Prepare the data for tokenizer training
    def get_training_corpus():
        for i in range(0, len(train_texts), 1000):
            yield train_texts[i: i + 1000]

    # Initialize and train the tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.BPEDecoder()
    trainer = trainers.BpeTrainer(vocab_size=wandb.config.vocab_size, show_progress=True)
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    tokenizer.save(tokenizer_path)

    # Save processed dataset to disk
    with open(dataset_path, "w") as f:
        json.dump({"train": train_texts, "val": val_texts}, f)
else:
    # Load processed dataset from disk
    with open(dataset_path, "r") as f:
        data = json.load(f)
        train_texts = data["train"]
        val_texts = data["val"]

# Load the trained tokenizer
tokenizer = Tokenizer.from_file(tokenizer_path)

def cycle(loader):
    while True:
        for data in loader:
            yield data

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        while True:
            # Preprocess text
            text = self.texts[idx]
            if len(text) > self.max_length:
                text = text[:self.max_length]
            
            encodings = self.tokenizer.encode(text)
            input_ids = encodings.ids
            
            # Pad the sequences if necessary
            if len(input_ids) < self.max_length:
                input_ids = input_ids + [0] * (self.max_length - len(input_ids) + 1)
            
            # Check if the input_ids are not all zeros
            if any(input_id != 0 for input_id in input_ids):
                return torch.tensor(input_ids, dtype=torch.long).to(device)
            
            # If input_ids are all zeros, pick another index
            idx = (idx + 1) % len(self.texts)

# Prepare the data
train_dataset = TextDataset(train_texts, tokenizer)
val_dataset = TextDataset(val_texts, tokenizer)
train_loader = cycle(DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True))
val_loader = cycle(DataLoader(val_dataset, batch_size=wandb.config.batch_size))

# Initialize the model
model = Griffin(GriffinConfig).to(device)

# Training parameters
learning_rate = wandb.config.learning_rate
gradient_accumulate_every = wandb.config.gradient_accumulate_every
validate_every = wandb.config.validate_every
checkpoint_every = 500

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
# optimizer = GaLoreAdamW(model.parameters(), lr=learning_rate)

# Initialize wandb logging
wandb.watch(model)

# Training loop
try:
    for i in tqdm.tqdm(range(wandb.config.num_batches), mininterval=10.0, desc="training"):
        model.train()
        accum_loss = 0

        for _ in range(gradient_accumulate_every):
            batch = next(train_loader)
            inputs, targets = batch[:, :-1], batch[:, 1:].reshape(-1)
            outputs = model(inputs)
            outputs = outputs.view(-1, wandb.config.vocab_size)
            loss = criterion(outputs, targets)
            loss.backward()
            accum_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad()

        avg_loss = accum_loss / gradient_accumulate_every
        print(f"Training loss at batch {i}: {avg_loss}")
        wandb.log({"training_loss": avg_loss, "batch": i})

        if i % validate_every == 0:
            model.eval()
            with torch.no_grad():
                val_batch = next(val_loader)
                val_inputs, val_targets = val_batch[:, :-1], val_batch[:, 1:].reshape(-1)
                val_outputs = model(val_inputs)
                val_outputs = val_outputs.view(-1, wandb.config.vocab_size)
                val_loss = criterion(val_outputs, val_targets)
                print(f"Validation loss at batch {i}: {val_loss.item()}")
                wandb.log({"validation_loss": val_loss.item(), "batch": i})

        # Save checkpoint
        if i != 0 and i % checkpoint_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"batch_{i}.pt")
            torch.save(model.state_dict(), checkpoint_path)

except KeyboardInterrupt:
    print("Training interrupted.")
finally:
    wandb.finish()

print("Training completed.")