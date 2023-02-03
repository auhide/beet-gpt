from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from lib.config import VOCAB_SIZE, CORPUS_PATH, TOKENIZER_PATH


tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# Choosing a whitespace pre-tokenization.
tokenizer.pre_tokenizer = Whitespace()

SPECIAL_TOKENS = [
    "[UNK]",
    "[LNE]",
    "[CHR]",
    "[NXT]",
    "[PAD]",
]

trainer = BpeTrainer(special_tokens=SPECIAL_TOKENS, vocab_size=VOCAB_SIZE)
tokenizer.train(
    files=[CORPUS_PATH], 
    trainer=trainer
)

tokens = tokenizer.encode("[LNE] I am an amazing assistant to the regional manager! [CHR] Dwight [NXT] Michael").tokens
print(f"Sample tokens: {tokens}")

tokenizer.save(TOKENIZER_PATH)
print("Tokenizer Saved!")
