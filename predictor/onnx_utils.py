from __future__ import annotations
from pathlib import Path
import os
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort


BASE_DIR = Path(__file__).resolve().parent.parent

# Per try.py: columns_to_use in this exact order
FEATURE_ORDER: List[str] = [
    'age', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'gender', 'BMI', 'height',
    'smoke', 'alco', 'active'
]

# Display + validation schema for UI and API
FEATURE_SCHEMA: Dict[str, Dict] = {
    'age':        {'label': 'Age (years)', 'min': 1,   'max': 120, 'step': 1,   'type': 'int'},
    'weight':     {'label': 'Weight (kg)', 'min': 20,  'max': 300, 'step': 0.1, 'type': 'float'},
    'ap_hi':      {'label': 'Systolic BP (mmHg)', 'min': 70,  'max': 250, 'step': 1,   'type': 'int'},
    'ap_lo':      {'label': 'Diastolic BP (mmHg)', 'min': 40, 'max': 150, 'step': 1,   'type': 'int'},
    'cholesterol':{'label': 'Cholesterol', 'type': 'int', 'widget': 'select', 'choices': [
        {'value': 1, 'label': '1 — normal'},
        {'value': 2, 'label': '2 — above normal'},
        {'value': 3, 'label': '3 — well above normal'},
    ], 'min': 1, 'max': 3, 'step': 1},
    'gluc':       {'label': 'Glucose', 'type': 'int', 'widget': 'select', 'choices': [
        {'value': 1, 'label': '1 — normal'},
        {'value': 2, 'label': '2 — above normal'},
        {'value': 3, 'label': '3 — well above normal'},
    ], 'min': 1, 'max': 3, 'step': 1},
    'gender':     {'label': 'Gender', 'type': 'int', 'widget': 'select', 'choices': [
        {'value': 1, 'label': 'Female'},
        {'value': 2, 'label': 'Male'},
    ], 'min': 1, 'max': 2, 'step': 1},
    'BMI':        {'label': 'BMI (auto-calculated)', 'min': 10, 'max': 60, 'step': 0.1, 'type': 'float', 'readonly': True},
    'height':     {'label': 'Height (cm)', 'min': 100, 'max': 250, 'step': 1, 'type': 'int'},
    'smoke':      {'label': 'Smoker', 'type': 'int', 'widget': 'select', 'choices': [
        {'value': 0, 'label': 'No'}, {'value': 1, 'label': 'Yes'}
    ], 'min': 0, 'max': 1, 'step': 1},
    'alco':       {'label': 'Alcohol use', 'type': 'int', 'widget': 'select', 'choices': [
        {'value': 0, 'label': 'No'}, {'value': 1, 'label': 'Yes'}
    ], 'min': 0, 'max': 1, 'step': 1},
    'active':     {'label': 'Physically active', 'type': 'int', 'widget': 'select', 'choices': [
        {'value': 0, 'label': 'No'}, {'value': 1, 'label': 'Yes'}
    ], 'min': 0, 'max': 1, 'step': 1},
}

def validate_features(data: Dict[str, object]) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Validate incoming features against FEATURE_SCHEMA.
    Returns (casted_features, errors). If errors is non-empty, validation failed.
    """
    casted: Dict[str, float] = {}
    errors: Dict[str, str] = {}
    # Pre-compute BMI if missing or invalid but height and weight present
    if ('BMI' not in data or str(data.get('BMI')).strip() == '' ) and ('height' in data and 'weight' in data):
        try:
            h = float(data['height']); w = float(data['weight'])
            if h > 0:
                data = dict(data)
                data['BMI'] = w / ((h/100.0) ** 2)
        except Exception:
            pass

    for k in FEATURE_ORDER:
        if k not in data:
            errors[k] = 'Missing value'
            continue
        sch = FEATURE_SCHEMA.get(k, {})
        v = data[k]
        try:
            val = float(v)
        except Exception:
            # Allow auto BMI compute if height/weight supplied but BMI invalid
            if k == 'BMI' and ('height' in data and 'weight' in data):
                try:
                    h = float(data['height']); w = float(data['weight'])
                    if h > 0:
                        val = w / ((h/100.0) ** 2)
                    else:
                        raise ValueError('height must be > 0')
                except Exception:
                    errors[k] = 'Not a number'
                    continue
            else:
                errors[k] = 'Not a number'
                continue
            continue
        # If select with limited choices, enforce membership
        if sch.get('widget') == 'select':
            allowed = {float(c['value']) for c in sch.get('choices', [])}
            if val not in allowed:
                errors[k] = 'Invalid choice'
                continue
        # Range check (kept for numeric bounds)
        mn = sch.get('min'); mx = sch.get('max')
        if mn is not None and val < mn:
            errors[k] = f'Value below minimum {mn}'
            continue
        if mx is not None and val > mx:
            errors[k] = f'Value above maximum {mx}'
            continue
        # Integer enforcement
        if sch.get('type') == 'int' and abs(val - round(val)) > 1e-6:
            errors[k] = 'Must be an integer'
            continue
        casted[k] = float(round(val) if sch.get('type') == 'int' else val)
    return casted, errors


class HeartONNX:
    def __init__(self, model_filename: str = 'final.onnx'):
        # Resolve model path with env override and fallback filenames
        env_file = os.environ.get('HEART_MODEL_PATH')
        env_path = Path(env_file).resolve() if env_file else None
        candidates = [
            BASE_DIR / model_filename,
            BASE_DIR / 'heart_model.onnx',
            BASE_DIR / 'final.onnx',
        ]
        if env_path and env_path.exists():
            self.model_path = env_path
        else:
            existing = next((p for p in candidates if p.exists()), None)
            self.model_path = existing or candidates[0]
        self.session = None
        self.input_names: List[str] = []
        self.output_names: List[str] = []

    def ensure_loaded(self) -> None:
        if self.session is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        providers = ort.get_available_providers()
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

    def build_input(self, features: Dict[str, float]) -> Dict[str, np.ndarray]:
        # Ensure BMI is consistent (auto-calc if possible)
        if 'BMI' in features and ('height' in features and 'weight' in features):
            h = float(features['height']); w = float(features['weight'])
            if h > 0:
                bmi_calc = w / ((h / 100.0) ** 2)
                # If user-supplied BMI is out of allowed range, replace with calculated
                if not (FEATURE_SCHEMA['BMI']['min'] <= features['BMI'] <= FEATURE_SCHEMA['BMI']['max']):
                    features['BMI'] = float(bmi_calc)
        # Build ONNX feed dict
        feed: Dict[str, np.ndarray] = {}
        for col in FEATURE_ORDER:
            if col not in features:
                raise ValueError(f"Missing feature: {col}")
            val = np.asarray([[float(features[col])]], dtype=np.float32)
            feed[col] = val
        return feed

    def predict(self, features: Dict[str, float]) -> Tuple[int, float]:
        self.ensure_loaded()
        feed = self.build_input(features)
        outputs = self.session.run(None, feed)
        # Support models with 2 outputs (prediction, probability) or 3 (scaled, prediction, probability)
        if len(outputs) == 2:
            prediction = outputs[0][0]
            probabilities = outputs[1][0]
        elif len(outputs) >= 3:
            prediction = outputs[1][0]
            probabilities = outputs[2][0]
        else:
            raise RuntimeError("Unexpected ONNX outputs: expected 2 or 3 tensors")
        pred_class = int(prediction)
        # Confidence is probability of predicted class from array
        conf = float(probabilities[pred_class])
        # Clamp to [0,1]
        conf = max(0.0, min(1.0, conf))
        return pred_class, conf


def label_text(y: int) -> str:
    return "Likely to have cardiovascular disease" if y == 1 else "Likely healthy"


def recommendation_for(p: float) -> str:
    pct = p * 100.0
    if pct >= 75.0:
        return (
            "High risk: Reduce sodium and cholesterol; avoid smoking and alcohol; begin daily 30-min moderate cardio; monitor BP/HR daily; see a cardiologist promptly."
        )
    if pct >= 50.0:
        return (
            "Moderate risk: Eat fruits/vegetables/fiber; limit processed food and sugar; exercise 20–30 min, 5x/week; track BP weekly; follow-up in 3–6 months."
        )
    return (
        "Low risk: Maintain balanced diet and exercise; keep healthy weight; routine checkups once/twice a year; monitor BP occasionally."
    )
