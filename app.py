from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("pkl/spam_classifier_model.pkl")
vectorizer = joblib.load("pkl/tfidf_vectorizer.pkl")

class Message(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"msg": "Welcome to Spamly API!"}

@app.post("/predict")
def predict(message: Message):
    vec = vectorizer.transform([message.text])
    pred = model.predict(vec)[0]
    return {"prediction": "SPAM" if pred == 1 else "NOT SPAM"}
