import os

from transformers import AutoTokenizer

from lib.config import BASE_MODEL, TOKENIZER_PATH, SPECIAL_TOKENS


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

special_tokens_dict = {'additional_special_tokens': SPECIAL_TOKENS}

print("Adding more special tokens...")
tokenizer.add_special_tokens(special_tokens_dict)

if not os.path.isdir(TOKENIZER_PATH):
    os.mkdir(TOKENIZER_PATH)
    
tokenizer.save_pretrained(TOKENIZER_PATH)
print("Tokenizer has been created and saved!")