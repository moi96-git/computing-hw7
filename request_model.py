import json
import requests

URL = "http://localhost:8000/predict"

# JSON Example 
with open("model_inputs.json", encoding="utf-8") as f:
    payload = json.load(f)

try:
    response = requests.post(URL, json=payload, timeout=10)
    response.raise_for_status()
    result = response.json()
    verdict = "Match" if result["match"] == 1 else "No match"
    print(f"Prediction: {verdict} (probability of a match: {result['match_probability']:.1%})")
# Server not running or wrong address
except requests.exceptions.ConnectionError as e:
    print("API not reachable. Is uvicorn running?", e)
# Server too slow to answer
except requests.exceptions.Timeout as e:
    print("The API took too long to answer:", e)
# Error status: 422 for invalid input or 500 for a missing model
except requests.exceptions.HTTPError as e:
    print("The API returned an error:", e.response.status_code, e.response.text)
# Response body was not JSON
except requests.exceptions.JSONDecodeError as e:
    print("The response was not valid JSON:", e)
# Any other requests problem (must come last: it is the parent class)
except requests.exceptions.RequestException as e:
    print("Other request error:", e)
