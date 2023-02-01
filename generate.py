import torch
from tokenizers import Tokenizer

from config import VOCAB_SIZE, MODEL_PATH, TOKENIZER_PATH
from inference import generate_response
from models import BeetGpt


# Loading the trained model.
model = BeetGpt(vocab_size=VOCAB_SIZE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
print("Model loaded!")


tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
print("Tokenizer loaded!")


# Generating response.
line = "Do you want some chocolate?"
character1 = "Michael"
character2 = "Pam"

print(f"{character1}: {line}")
response = generate_response(
    model=model, 
    tokenizer=tokenizer,
    line=line, 
    character=character1, next_character=character2,
    device="cpu"
)
print(f"{character2}: {response}")