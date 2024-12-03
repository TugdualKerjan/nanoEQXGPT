from tokenizers.trainers import BpeTrainer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset

# Initialize the tokenizer and trainer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(vocab_size=512, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.pre_tokenizer = Whitespace()

# Load the dataset
dataset = load_dataset("roneneldan/TinyStories", split="train")

# Extract text data
def get_texts(dataset):
    for item in dataset:
        yield item["text"]  # Adjust the key to match your dataset structure

# Train the tokenizer
tokenizer.train_from_iterator(get_texts(dataset), trainer)

# Save the tokenizer
tokenizer.save("data/tokenizer-tinystories.json")