# The Office Dialogue Generator

<br>

<p align="center">
<img src="./static/the-office-logo.jpg"></center>
</p>

<a id="webpage"></a>
## Webpage
Play around with the model on [this page](https://beet-gpt.streamlit.app/).

<a id="model"></a>
## Model
[![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/auhide/gpt2-the-office-dialogue-generation)

The model is GPT-2, that's fine-tuned on the whole transcript of The Office (seasons 1-9) with a causal objective (autoregressively).

[Download BeetGPT's Weights](https://drive.google.com/file/d/1--Ts8_8Z2K1zH7VS2KnLksQBEb7mFx2c/view?usp=sharing)

<a id="dataset"></a>
## Dataset
The model has been trained on all 9 seasons of The Office for a single epoch.
The dataset can be found on [Kaggle](https://www.kaggle.com/datasets/fabriziocominetti/the-office-lines/code?datasetId=1807639).

## Examples
Line '*Good morning!*', said by *Michael Scott*:
> Michael: Good morning!

> Dwight: Hello everyone! Remember, our wedding’s gonna be wonderful.
___
> Michael: Good morning!

> Pam: Good morning, Michael.
___
> Michael: Good morning!

> Angela:  All right, go away!
___
> Micheal: Good morning!

> Creed: Hello. [Michael approaches table] Are you Michael Scott?

<br>

Line '*That's what she said!*', said by *Michael Scott*:
> Michael: That's what she said!

> Dwight: Ha! I don’t get it.
---
> Michael: That's what she said!

> Pam: Am I going to tell them that?
___
> Michael: That's what she said!

> Angela: \[leaves the party\]
---
> Michael: That's what she said!

> Creed: \[Michael is reading a book to a hanging tire\] Wow.