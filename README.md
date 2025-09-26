# Heart Risk Predictor (Django + ONNX)

Minimal, stylish, single-page Django app that predicts cardiovascular disease risk using your existing ONNX model and returns specific, actionable recommendations based on confidence.

This project is live hosted at Render - https://heart-disease-prediction-using-pyspark.onrender.com/
## What this app does
- Single-page UI with a clean card layout
- Sends your inputs as JSON to `/api/predict` and renders the result dynamically
- Loads ONNX model with `onnxruntime`
- Confidence visualized with a progress bar and tailored recommendations

## Requirements
- Virtual environment `myenv` is already activated
- Model file present at project root: `cardio_rf_pipeline_model.onnx`

## Install dependencies (inside your activated venv)
```powershell
pip install -r requirements.txt
```

## Run the app (dev)
```powershell
python manage.py runserver
```
Open: http://127.0.0.1:8000

## Model I/O (based on try.py)
- Feature order (exact):
  - age, weight, ap_hi, ap_lo, cholesterol, gender, BMI, height
- Input to ONNX: a dict mapping each feature name to a float32 array of shape (1,1)
- Outputs: `[scaled_features, prediction, probability]`
  - `prediction`: integer class 0/1
  - `probability`: array of class probabilities; confidence = probability[prediction]

## Example API request body
See `test_input.json`:
```json
{
  "age": 30.0,
  "weight": 60.0,
  "ap_hi": 120,
  "ap_lo": 80,
  "cholesterol": 1,
  "gender": 1,
  "BMI": 20.76,
  "height": 170
}
```

## Test via PowerShell
```powershell
$body = Get-Content -Raw -Path .\test_input.json
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/predict -Method Post -ContentType 'application/json' -Body $body
```

## Expected JSON response (shape)
```json
{
  "label": "Likely healthy",
  "confidence": 83.7,
  "recommendation": "Low risk: Maintain balanced diet and exercise; keep healthy weight; routine checkups once/twice a year; monitor BP occasionally."
}
```

## Notes
- Demo only; no PHI; not medical advice.
- SQLite is unused; kept for default Django setup.
