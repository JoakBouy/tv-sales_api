import nest_asyncio
nest_asyncio.apply()
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.openapi.utils import get_openapi

# Load the saved model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

app = FastAPI(
    title="Sales Prediction API",
    description="API for predicting sales based on TV marketing expenses.",
    version="1.0.0",
)

class TVMarketingInput(BaseModel):
    tv: float

@app.post("/predict", summary="Predict sales based on TV marketing expenses")
def predict(tv_input: TVMarketingInput):
    tv = tv_input.tv
    sales_pred = model.predict([[tv]])
    predicted_sales = sales_pred.item()  # Extract the scalar value using item()
    return {"predicted_sales": predicted_sales}


@app.get("/openapi.json", include_in_schema=False)
def get_open_api_endpoint():
    return app.openapi()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)