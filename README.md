# Exercise 7: Python APIs

This repository is intended to contain the final version of the assignment for Exercise 7: Python APIs.

The final project includes a small machine learning API built with FastAPI, where a trained model is loaded from disk and used to generate predictions from input data sent to an endpoint.

## What the repository contains

- A trained machine learning model serialized with `joblib` (for example, a `.joblib` file)
- A Python file that starts the API server and defines the prediction endpoint
- A sample JSON file showing the format of the input features required by the model
- A Python client script that sends an HTTP request to the API and prints the prediction
- Optional model-training code used to generate the saved model
- A minimal setup section describing dependencies and how to run the service locally

## Expected final structure

```text
computing-hw7/
├── README.md
├── model.joblib
├── api_app.py
├── example_input.json
├── request_model.py
├── requirements.txt   # optional, if used
└── train_model.py    # optional, if included for reproducibility
```

## Purpose of the project

The goal is to demonstrate how a trained machine learning model can be exposed through an API so that external programs can submit a feature vector or JSON payload and receive a prediction in return.

The final version should allow the user to:

1. Train or load a machine learning model
2. Run the API locally with Uvicorn
3. Send a request to the server
4. Receive a prediction from the model
5. Handle common API-client issues such as connection errors or invalid input

## Typical local run flow

```bash
uvicorn api_app:app --reload
```

Then run the client script in a second terminal:

```bash
python request_model.py
```

The API will usually be served at:

```text
http://localhost:8000
