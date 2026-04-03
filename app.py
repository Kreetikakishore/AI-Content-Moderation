from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model/moderation_model.pkl")

class TextInput(BaseModel):
    text: str

@app.post("/analyze")
def analyze(input: TextInput):
    text = input.text
    prediction = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    
    toxic_score = float(proba[1])
    
    # Severity scoring (upgrade idea from the post!)
    if toxic_score < 0.3:
        severity = "safe"
        level = 0
    elif toxic_score < 0.5:
        severity = "low"
        level = 1
    elif toxic_score < 0.75:
        severity = "medium"
        level = 2
    else:
        severity = "high"
        level = 3

    return {
        "is_toxic": bool(prediction),
        "toxic_score": round(toxic_score * 100, 1),
        "severity": severity,
        "severity_level": level,
    }

@app.get("/")
def root():
    return {"status": "AI Content Moderation API is running"}