import os
import re

import gdown
import torch
import streamlit as st
from tokenizers import Tokenizer

from models import BeetGpt
from inference import generate_response
from config import (
    VOCAB_SIZE, MODEL_PATH, TOKENIZER_PATH,
    FROM_CHARACTERS, TO_CHARACTERS, MODEL_URL
)


# Firstly we will start by downloading the model if it has not been downloaded yet.
current_dir = os.getcwd()
print(f"Current directory: {current_dir}")

if MODEL_PATH not in os.listdir(current_dir):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

else:
    print("The model weights already exist! No need to download them!")


st.set_page_config(
    page_title="Beet GPT2",
)


@st.cache(show_spinner=False)
def get_model():
    model = BeetGpt(vocab_size=VOCAB_SIZE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    return model


def _format_response(response):
    response = re.sub(r"\s+([\.?!\,\;\:'\"]+)", r"\1", response)
    response = re.sub(r"\s+(’)\s+", r"\1", response)

    return response


st.title("The Office dialogue generation")

col1, col2 = st.columns(2)
from_character, to_character = "", ""


with st.spinner("Loading model..."):
    model = get_model()

with st.spinner("Loading tokenizer..."):
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)


with col1:
    from_character = st.selectbox(
        'Line from',
        FROM_CHARACTERS
    )

with col2:
    to_character = st.selectbox(
        'Answer from',
        TO_CHARACTERS
    )

line = st.text_input(label="", placeholder="Enter line here.")

if st.button("Generate response") or line:
    if line != "":
        with st.spinner("Generating response..."):
            response = generate_response(
                model=model, 
                tokenizer=tokenizer,
                line=line,
                character=from_character,
                next_character=to_character
            )
            st.write(f"{from_character}: {line}")
            st.write(f"{to_character}: {_format_response(response)}")
    else:
        st.warning('Please, enter a line!')
