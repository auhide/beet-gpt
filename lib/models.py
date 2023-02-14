from torch import nn
from transformers import AutoModel

from lib.config import BASE_MODEL


class BeetGpt(nn.Module):
    
    def __init__(self, vocab_size, return_attentions=False):
        super().__init__()
        
        self.return_attentions = return_attentions
        self.base_model = AutoModel.from_pretrained(BASE_MODEL)
        self.base_model.resize_token_embeddings(vocab_size)
        
        self.project = nn.Linear(768, vocab_size)
    
    def forward(self, input_ids):
        out = self.base_model(input_ids, output_attentions=self.return_attentions)
        
        hidden_state = out.last_hidden_state
        # hidden_state shape: (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        output = self.project(hidden_state)
        # output shape: (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        
        return output, out.attentions
