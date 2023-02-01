VOCAB_SIZE = 30_000
SPECIAL_TOKENS = [
    "[UNK]",
    "[LNE]",
    "[CHR]",
    "[NXT]",
    "[PAD]",
]
BASE_MODEL = "gpt2"
MODEL_PATH = ".beet-gpt2.pt"
MODEL_URL = 'https://drive.google.com/uc?id=1--Ts8_8Z2K1zH7VS2KnLksQBEb7mFx2c'
CORPUS_PATH = "corpus.txt"
TOKENIZER_PATH = ".beetokenizer.pt"

# Streamlit configuration variables:
FROM_CHARACTERS = [
    "Michael", 
    "Dwight", 
    "Jim", 
    "Pam",
    "Kevin",
    "Angela",
    "Andy",
    "Meredith",
    "Creed",
    "Stanley",
    "Phyllis",
    "Oscar",
    "Toby",
    "Kelly",
    "Erin",
    "Darryl",
]
TO_CHARACTERS = FROM_CHARACTERS[1:] + [FROM_CHARACTERS[0]]
