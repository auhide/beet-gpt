import re

import torch
from torch import nn


def generate_response(
    model: nn.Module, 
    tokenizer,
    line: str, 
    character: str, next_character: str,
    device="cpu"
):
    line = f"[LNE] {line} [CHR] {character} [NXT] {next_character}"
    line = tokenizer.encode(line).ids
    predictions = line
    line = torch.LongTensor(line).unsqueeze(0).to(device)
    
    model.eval()

    with torch.no_grad():
        curr_token_pred = ""
        
        while curr_token_pred != "[CHR]":
            pred, _ = model(line)
            argmax_pred = pred.argmax(-1).squeeze(0)[-1]
            predictions.append(int(argmax_pred))
            curr_token_pred = tokenizer.decode(
                [argmax_pred], 
                skip_special_tokens=False
            )
            line = torch.LongTensor(predictions).unsqueeze(0).to(device)

    model.train()
    
    response = tokenizer.decode(predictions, skip_special_tokens=False)
    response = re.findall(r"\[LNE\]\s(.+?)\[CHR\]", response)[-1]
    
    return response
