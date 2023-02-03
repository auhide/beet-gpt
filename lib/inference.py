import re

import torch
from torch import nn


def generate_response(
    model: nn.Module, 
    tokenizer,
    line: str,
    k: int,
    device="cpu"
):
    line = tokenizer.encode(line).ids
    predictions = line
    line = torch.LongTensor(line).unsqueeze(0).to(device)
    
    model.eval()

    with torch.no_grad():
        curr_token_pred = ""
        
        while curr_token_pred != "[CHR]":
            pred, _ = model(line)
            topk = torch.topk(pred, k=k)[-1]

            argmax_pred = topk[:, :, -1].squeeze(0)[-1]
            predictions.append(int(argmax_pred))
            curr_token_pred = tokenizer.decode(
                [argmax_pred], 
                skip_special_tokens=False
            )
            line = torch.LongTensor(predictions).unsqueeze(0).to(device)

    model.train()
    
    full_response = tokenizer.decode(predictions, skip_special_tokens=False)
    final_answer = re.findall(r"\[LNE\]\s(.+?)\[CHR\]", full_response)[-1]
    
    return final_answer
