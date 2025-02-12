from transformer import *
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from datasets import load_dataset
from tqdm import tqdm

# Load the Multi30k dataset
ds = load_dataset("bentrevett/multi30k")

# Get split data
train_data = ds['train']
val_data = ds['validation']
test_data = ds['test']

# Dataset needs to be formatted so -> train_data[src_lang], train_data[trg_lang]
src_lang = 'en'
trg_lang = 'de'

# Add special tokens
SPECIAL_TOKENS = ['<bos>','<eos>','<pad>','<unk>']

# Declare English & German character sets
src_chars = set()
trg_chars = set()

src_sentences = train_data[src_lang]
trg_sentences = train_data[trg_lang]

# Process all sentences in English to get all Chars
for sentence in src_sentences:
    src_chars.update(sentence)

# Process all sentences in German to get all Chars
for sentence in trg_sentences:
    trg_chars.update(sentence)

# Char sets -> sorted list
src_chars = sorted(src_chars)
trg_chars = sorted(trg_chars)

# Char index + special tokens
ind_trg = {k:v for k,v in enumerate(SPECIAL_TOKENS+trg_chars)}
trg_ind = {v:k for k,v in enumerate(SPECIAL_TOKENS+trg_chars)}
ind_src = {k:v for k,v in enumerate(SPECIAL_TOKENS+src_chars)}
src_ind = {v:k for k,v in enumerate(SPECIAL_TOKENS+src_chars)}


########################################### LIMIT SENTENCE SIZE ###################################################
max_sequence_length = 128 # paper: 512

def is_valid_tokens(sentence, vocab):
    for token in list(set(sentence)):
        if token not in vocab:
            return False
    return True

def is_valid_length(sentence, max_sequence_length):
    return len(list(sentence)) < (max_sequence_length - 1)

valid_sentence_indices = []
for i in range(len(trg_sentences)):
    trg_sentence, src_sentence = trg_sentences[i], src_sentences[i]
    if is_valid_length(trg_sentence, max_sequence_length) \
        and is_valid_length(src_sentence, max_sequence_length) \
        and is_valid_tokens(trg_sentence, trg_chars):
            valid_sentence_indices.append(i)

print(f"Number of sentences: {len(trg_sentences)}")
print(f"Number of VALID sentences: {len(valid_sentence_indices)}\n")

# After computing valid_sentence_indices, filter the sentences
filtered_src_sentences = [src_sentences[i] for i in valid_sentence_indices]
filtered_trg_sentences = [trg_sentences[i] for i in valid_sentence_indices]
##################################################################################################################


class TranslationDataset(Dataset):
    def __init__(self, src_sentences, trg_sentences):
        self.src = src_sentences
        self.trg = trg_sentences

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.trg[idx]

# Initialize DataLoader
dataset = TranslationDataset(filtered_src_sentences, filtered_trg_sentences)

batch_size = 8 # Number of sentences per iteration
dataloader = DataLoader(dataset, batch_size=batch_size)
iterator = iter(dataloader)

# Sample iteration of dataset
# print("Batches example: ")
# for batch_num, batch in enumerate(iterator):
#     print(f"{batch}\n")
#     if batch_num > 3:
#         break


############################################### TOKENIZATION #####################################################
def tokenize(sentence, language_to_index, bos=True, eos=True):
    sentence_word_indices = []
    for token in list(sentence):
        if token in language_to_index:
            sentence_word_indices.append(language_to_index[token])
        else:
            sentence_word_indices.append(language_to_index['<unk>'])
    if bos:
        sentence_word_indices.insert(0, language_to_index['<bos>'])
    if eos:
        sentence_word_indices.append(language_to_index['<eos>'])
    for _ in range(len(sentence_word_indices), max_sequence_length):
        sentence_word_indices.append(language_to_index['<pad>'])
    return torch.tensor(sentence_word_indices)

# src_tokenized = []
# trg_tokenized = []

# for sentence_num in range(batch_size):
#     src_sentence, trg_sentence = batch[0][sentence_num], batch[1][sentence_num]
#     src_tokenized.append(tokenize(src_sentence, src_ind))
#     trg_tokenized.append(tokenize(trg_sentence, trg_ind))

# src_tokenized = torch.stack(src_tokenized)
# trg_tokenized = torch.stack(trg_tokenized)

# print(trg_tokenized)
##################################################################################################################


# Define the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the Transformer model
src_vocab_size = len(src_ind)  # Vocabulary size for source language
trg_vocab_size = len(trg_ind)  # Vocabulary size for target language
src_pad_idx = src_ind['<pad>']  # Padding index for source language
trg_pad_idx = trg_ind['<pad>']  # Padding index for target language

model = Transformer(
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    trg_pad_idx,
    embed_size=256, # paper: 512 or 256
    num_layers=4, # paper: 6
    forward_expansion=2, # paper: 4
    heads=8, # paper: 8
    dropout=0.1, # paper: 0.1
    device=device,
    max_length=max_sequence_length # paper: 512
).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)  # Ignore padding index
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 200 # Change as you see fit
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    # Wrap the dataloader with tqdm for a progress bar
    with tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
        for batch_idx, batch in enumerate(pbar):
            src_sentences, trg_sentences = batch

            # Tokenize the sentences and pad them to the same length
            src_tokenized = []
            trg_tokenized = []
            for src_sentence, trg_sentence in zip(src_sentences, trg_sentences):
                # Tokenize and truncate if necessary
                src_tokenized.append(tokenize(src_sentence, src_ind, bos=True, eos=True))
                trg_tokenized.append(tokenize(trg_sentence, trg_ind, bos=True, eos=True))

            # Pad sequences to the same length in the batch
            src_tokenized = pad_sequence(src_tokenized, batch_first=True, padding_value=src_pad_idx)
            trg_tokenized = pad_sequence(trg_tokenized, batch_first=True, padding_value=trg_pad_idx)

            # Check for invalid indices
            if (src_tokenized >= src_vocab_size).any() or (trg_tokenized >= trg_vocab_size).any():
                pbar.write(f"Invalid indices found. Skipping batch.")
                continue

            src_tokenized = src_tokenized.to(device)
            trg_tokenized = trg_tokenized.to(device)

            # Forward pass
            output = model(src_tokenized, trg_tokenized[:, :-1])  # Exclude the last token from the target

            # Reshape the output and target for loss calculation
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg_tokenized[:, 1:].contiguous().view(-1)  # Exclude the first token from the target

            # Compute the loss
            loss = criterion(output, trg)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Update the progress bar with the current loss
            pbar.set_postfix({"Loss": loss.item()})

    # Print the average loss for the epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {epoch_loss / len(dataloader)}")

# Save the model after training
torch.save(model.state_dict(), "transformer_model.pth")

