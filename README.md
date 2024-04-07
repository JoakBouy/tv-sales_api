# Sales Prediction API
This is a FastAPI-based backend service that provides an API for predicting sales based on TV marketing expenses.

# Features
- Exposes a /predict endpoint to receive TV marketing expenses and return the predicted sales.
- Utilizes a pre-trained machine learning model to perform the sales prediction.
- Includes CORS (Cross-Origin Resource Sharing) middleware to allow access from any origin (for development purposes, adjust this for production).
- Provides an OpenAPI endpoint (/openapi.json) to generate the API documentation.

# Prerequisites
- Python 3.7 or higher
- FastAPI
- Uvicorn (for running the server)
- Pydantic
- Pickle (for loading the pre-trained model)

# Installation
1. Clone the repository:
```
git clone https://github.com/JoakBouy/tv-sales_api.git
```
2. Create a virtual enviroment and activate it (optional, but recommended):
```
python -m venv venv
source venv/bin/activate
```
3. Install the required dependencies:
```
pip install -r requirements.txt
```
4. Ensure you have the pre-trained machine learning model saved as model.pkl in the same directory as the main.py file.

# Running the Server
1. Start the FastAPI server:
```
uvicorn app:app --reload
```
Or
```
python main.py
```
2. The server will start running at http://127.0.0.1:8000

# Endpoints
# POST /predict
- Description: Predict sales based on TV marketing expenses
- Request Body:
    - tv (float): TV marketing expenses.
- Response:
    - predicted_sales (float): The predicted sales based on the provided TV marketing expenses.

# GET /openapi.json
- Description: Retrieve the OpenAPI documentation for the API.

# Error Handling
The API handles exceptions and returns appropriate HTTP status codes:

  - 400 Bad Request: If there is an issue with the input data (e.g., invalid tv value).

# Future Improvements
- Implement more robust error handling and logging.
- Add authentication and authorization mechanisms for the API.
- Integrate the API with a database for storing and retrieving historical data.
- Deploy the API to a cloud platform for production use.
