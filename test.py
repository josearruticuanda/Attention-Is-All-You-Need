from transformer import *
from datasets import load_dataset

# Load the Multi30k dataset
ds = load_dataset("bentrevett/multi30k")

# Get training data
train_data = ds['train']

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

# Process all sentences in English and get all Chars
for sentence in src_sentences:
    src_chars.update(sentence)

# Process all sentences in German and get all Chars
for sentence in trg_sentences:
    trg_chars.update(sentence)

# Char sets -> sorted list
src_chars = sorted(src_chars)
trg_chars = sorted(trg_chars)

# Dictionary index / language to index & index to language
ind_trg = {k:v for k,v in enumerate(SPECIAL_TOKENS+trg_chars)}
trg_ind = {v:k for k,v in enumerate(SPECIAL_TOKENS+trg_chars)}
ind_src = {k:v for k,v in enumerate(SPECIAL_TOKENS+src_chars)}
src_ind = {v:k for k,v in enumerate(SPECIAL_TOKENS+src_chars)}


def ensure_special_tokens(index_to_language, language_to_index):
    """Make sure special tokens are in both dictionaries"""
    for token in SPECIAL_TOKENS:
        if token not in language_to_index:
            # Find an unused index
            new_idx = max(index_to_language.keys()) + 1
            language_to_index[token] = new_idx
            index_to_language[new_idx] = token
    return index_to_language, language_to_index

# Ensure special tokens are in dictionaries
ind_trg, trg_ind = ensure_special_tokens(ind_trg, trg_ind)
ind_src, src_ind = ensure_special_tokens(ind_src, src_ind)


############################################### TOKENIZATION #####################################################
max_sequence_length = 128 # paper: 512

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
##################################################################################################################


############################################## DETOKENIZATION ####################################################
def detokenize(indices, index_to_language):
    tokens = []
    eos_idx = trg_ind.get('<eos>', -1)
    bos_idx = trg_ind.get('<bos>', -1)
    pad_idx = trg_ind.get('<pad>', -1)
    unk_idx = trg_ind.get('<unk>', -1)
    
    for idx in indices:
        if idx == eos_idx:
            break
        if idx not in [bos_idx, pad_idx, unk_idx]:
            if idx in index_to_language:
                tokens.append(index_to_language[idx])
            else:
                tokens.append('?')  # Unknown token
    return ''.join(tokens)
##################################################################################################################


################################################# TRANSLATE ######################################################
def translate_sentence(sentence, model, src_ind, trg_ind, ind_trg, device, max_length):
    model.eval()
    
    # Tokenize the input
    src_tokenized = tokenize(sentence, src_ind, bos=True, eos=True)
    src_tensor = src_tokenized.unsqueeze(0).to(device)  # Add batch dimension
    
    # Create an initial target tensor with just the <bos> token
    trg_indexes = [trg_ind['<bos>']]
    
    for _ in range(max_length):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(src_tensor, trg_tensor)
            
        pred_token = output.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token)
        
        if pred_token == trg_ind['<eos>']:
            break
    
    # Convert the indices back to a sentence
    translated_sentence = detokenize(trg_indexes, ind_trg)
    return translated_sentence
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


def main():
    # Make sure the model is in eval mode
    model.eval()
    
    # Load the trained model weights if needed
    try:
        model.load_state_dict(torch.load("transformer_model.pth", map_location=device))
        print("Loaded saved model weights.\n")
    except:
        print("Using model without loading weights. Make sure the model is trained.\n")
    
    # Start translation loop
    while True:
        input_sentence = input("\nEnter an English sentence to translate to German (or 'q' to quit): ")
        
        if input_sentence.lower() == 'q':
            break
            
        translated_sentence = translate_sentence(
            input_sentence, model, src_ind, trg_ind, ind_trg, 
            device, max_sequence_length
        )
        
        print(f"Translation: {translated_sentence}")

if __name__ == "__main__":
    main()