from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import tensorflow as tf
from .data_preprocessing import preprocessing_stage
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

MAX_LEN = 77

app = FastAPI()


class InferenceRequest(BaseModel):
    text: str


class Token(BaseModel):
    value: str
    type: str


class InferenceResponse(BaseModel):
    tokens: list[Token]


class ModelMock:
    def predict(self, text):
        return [
            ["Elon", "Musk", "created", "SpaceX"],
            ["pers-b", "pers-i", "o", "comp-b"],
        ]


model = tf.keras.models.load_model('models/ner_model.h5')
tokenizer = None

with open('models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


@app.post("/inference")
async def inference(request: InferenceRequest) -> InferenceResponse:
    input_text = request.text
    preprocessed_text = preprocessing_stage(input_text)
    input_tokens = tokenizer.texts_to_sequences([preprocessed_text])[0]
    input_tokens = pad_sequences([input_tokens], maxlen=MAX_LEN, padding='post')
    token_values, token_types = model.predict(preprocessed_text)[0]

    return InferenceResponse(
        tokens=[
            Token(value=tok_value, type=tok_type)
            for tok_value, tok_type in zip(token_values, token_types, strict=True)
        ]
    )
