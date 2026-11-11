import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# One speed date: the 7 date-night ratings used in training, each on a 0-10 scale
class DateRatings(BaseModel):
    like: float = Field(ge=0, le=10, description="How much I liked my partner")
    attractive_partner: float = Field(ge=0, le=10, description="I rate my partner: attractiveness")
    funny_partner: float = Field(ge=0, le=10, description="I rate my partner: being funny")
    shared_interests_partner: float = Field(ge=0, le=10, description="I rate my partner: shared interests")
    guess_prob_liked: float = Field(ge=0, le=10, description="How likely I think my partner likes me")
    attractive_o: float = Field(ge=0, le=10, description="My partner rates me: attractiveness")
    funny_o: float = Field(ge=0, le=10, description="My partner rates me: being funny")


app = FastAPI(title="Speed Dating Match API")


@app.get("/")
async def root():
    return {"message": "API is running"}


# Predict whether a date ends in a match
@app.post("/predict")
async def predict(ratings: DateRatings):
    try:
        model = joblib.load("model_speed_dating.joblib")
        row = pd.DataFrame([ratings.model_dump()])
        prediction = model.predict(row)[0]
        probability = model.predict_proba(row)[0, 1]
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model file not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Model rejected the input: {e}")
    return {
        "match": int(prediction),
        "match_probability": round(float(probability), 3),
    }
