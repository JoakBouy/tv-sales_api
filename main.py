import nest_asyncio
nest_asyncio.apply()
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load the saved model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

app = FastAPI(
    title="Sales Prediction API",
    description="API for predicting sales based on TV marketing expenses.",
    version="1.0.0",
)

# Configure CORS to allow all origins for now 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

class TVMarketingInput(BaseModel):
    tv: float = Field(gt=0, description="TV marketing expenses")

@app.post("/predict", summary="Predict sales based on TV marketing expenses")
def predict(tv_input: TVMarketingInput):
    try:
        tv = tv_input.tv
        sales_pred = model.predict([[tv]])
        predicted_sales = sales_pred.item()
        return {"predicted_sales": predicted_sales}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/openapi.json", include_in_schema=False)
def get_open_api_endpoint():
    return app.openapi()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
