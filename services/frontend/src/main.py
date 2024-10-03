import streamlit as st
from annotated_text import annotated_text
import requests
from pydantic import BaseModel

class Token(BaseModel):
    entity: str
    index: int
    word: str
    start: int
    end: int

st.title("Named Entity Recognition")

text_input = st.text_input(
        "Enter some text (in English) 👇",
    )

# O 	Outside of a named entity
# B-MISC 	Beginning of a miscellaneous entity right after another miscellaneous entity
# I-MISC 	Miscellaneous entity
# B-PER 	Beginning of a person’s name right after another person’s name
# I-PER 	Person’s name
# B-ORG 	Beginning of an organization right after another organization
# I-ORG 	organization
# B-LOC 	Beginning of a location right after another location
# I-LOC 	Location

if text_input:
    response = requests.post("http://pmldl-backend:8000/inference", json={"text": text_input})
    response = response.json()
    prev_token = None
    annotation = []

    for r in response:
        token = Token(**r)
        if prev_token is None:
            annotation.append(text_input[0:token.start])
        elif abs(prev_token.end - token.start) > 1:
            annotation.append(text_input[prev_token.end:token.start])
        else:
            annotation[-1][0] += text_input[prev_token.end:token.start]

        if token.entity == "O":
            annotation.append(text_input[token.start:token.end])
        elif token.entity.startswith("B-") and (prev_token == None or abs(prev_token.end - token.start) >= 1):
            annotation.append([text_input[token.start:token.end], token.entity[2:]])
        elif token.entity.startswith("B-") or token.entity.startswith("I-"):
            if type(annotation[-1]) == list:
                annotation[-1] = [annotation[-1][0] + text_input[token.start:token.end], token.entity[2:]]
            else:
                annotation.append([text_input[token.start:token.end], token.entity[2:]])

        prev_token = token
        
    for i in range(len(annotation)):
        if type(annotation[i]) == list:
            annotation[i] = tuple(annotation[i])
    annotated_text(annotation)