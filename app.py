import os
import re

import gdown
import torch
import streamlit as st
from tokenizers import Tokenizer

from lib.models import BeetGpt
from lib.inference import generate_response
from lib.config import (
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


# Setting the initial session states of the page.
if "full_response" not in st.session_state:
    st.session_state.full_response = ""

if "curr_response" not in st.session_state:
    st.session_state.curr_response = ""


# Caching the model since otherwise it will be loaded on each page reload,
# which is cumbersome.
@st.cache(show_spinner=False)
def get_model():
    model = BeetGpt(vocab_size=VOCAB_SIZE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    return model


def _format_response(response: str) -> str:
    response = re.sub(r"\s+([\.?!\,\;\:'\"]+)", r"\1", response)
    response = re.sub(r"\s+(’)\s+", r"\1", response)

    return response


def _display_dialogue(raw_text: str):
    matches = re.findall(
        r"\[LNE\](.+?)\[CHR\](.+?)\[",
        string=raw_text
    )
    dialogue = []
    
    for line, character in matches:
        dialogue.append(
            _format_response(f"{character}: {line}")
        )

    for line in dialogue:
        st.write(line)


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

toggle_btn1, toggle_btn2, toggle_btn3 = st.columns(3)

with toggle_btn1:
    k = st.slider("Top K Answer", 1, 5, 1)

with toggle_btn3:
    st.markdown("##")
    dialogue_is_continuous = st.checkbox(label="Continuous dialogue")


line = st.text_input(label="", placeholder="Enter line here.")
st.session_state.curr_response = line


btn1_col1, btn1_col2, btn1_col3 = st.columns(3)

with btn1_col2:
    if st.button("Generate response"):
        if line != "":
            with st.spinner("Generating response..."):
                
                if dialogue_is_continuous:
                    st.session_state.full_response += f"[LNE] {st.session_state.curr_response} [CHR] {from_character} [NXT] {to_character} "
                else:
                    st.session_state.full_response = f"[LNE] {st.session_state.curr_response} [CHR] {from_character} [NXT] {to_character}"

                st.session_state.curr_response = generate_response(
                    model=model, 
                    tokenizer=tokenizer,
                    line=st.session_state.full_response,
                    k=k
                )

                
                st.session_state.full_response += f"[LNE] {st.session_state.curr_response} [CHR] {to_character} [NXT] {from_character} "
        else:
            st.warning('Please, enter a line!')

_display_dialogue(st.session_state.full_response)


clear_col1, clear_col2, clear_col3, clear_col4, clear_col5 = st.columns(5)

with clear_col3:
    if st.button("🗑️", help="Clear dialogue"):
        st.session_state.full_response = ""
