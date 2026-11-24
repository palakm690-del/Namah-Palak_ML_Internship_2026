"""Local web interface for the saved student placement Random Forest."""
from __future__ import annotations

from pathlib import Path
from threading import Lock

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
MODEL_PATH = PROJECT_DIR / "final_model.pkl"
app = Flask(__name__, static_folder=str(APP_DIR), static_url_path="")

try:
    model = joblib.load(MODEL_PATH)
    model_error = None
except Exception as exc:  # Surface a useful message in the UI if the artifact cannot load.
    model = None
    model_error = str(exc)

explainer = None
explainer_lock = Lock()

REQUIRED = {
    "Age": (18, 30),
    "CGPA": (0, 10),
    "Internships": (0, 10),
    "Projects": (0, 20),
    "Coding_Skills": (1, 10),
    "Communication_Skills": (1, 10),
    "Aptitude_Test_Score": (0, 100),
    "Soft_Skills_Rating": (1, 10),
    "Certifications": (0, 20),
    "Backlogs": (0, 10),
}
VALID_DEGREES = {"B.Tech", "BCA", "MCA", "B.Sc"}
VALID_BRANCHES = {"CSE", "Civil", "ECE", "IT", "ME"}
VALID_GENDERS = {"Female", "Male"}
DISPLAY_NAMES = {
    "Age": "Age", "Gender": "Gender", "CGPA": "CGPA", "Internships": "Internships",
    "Projects": "Projects", "Coding_Skills": "Coding skills",
    "Communication_Skills": "Communication skills", "Aptitude_Test_Score": "Aptitude score",
    "Soft_Skills_Rating": "Soft skills rating", "Certifications": "Certifications",
    "Backlogs": "Backlogs",
}
RECOMMENDABLE = {
    "CGPA": ("increase", 0.1, 10),
    "Internships": ("increase", 1, 10),
    "Projects": ("increase", 1, 20),
    "Coding_Skills": ("increase", 1, 10),
    "Communication_Skills": ("increase", 1, 10),
    "Aptitude_Test_Score": ("increase", 1, 100),
    "Soft_Skills_Rating": ("increase", 1, 10),
    "Certifications": ("increase", 1, 20),
    "Backlogs": ("decrease", 1, 0),
}


def _expected_columns() -> list[str]:
    if model is not None and hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    # Column order produced by the notebook's pandas get_dummies(drop_first=True).
    return [
        "Age", "Gender", "Degree_B.Tech", "Degree_BCA", "Degree_MCA",
        "Branch_Civil", "Branch_ECE", "Branch_IT", "Branch_ME", "CGPA",
        "Internships", "Projects", "Coding_Skills", "Communication_Skills",
        "Aptitude_Test_Score", "Soft_Skills_Rating", "Certifications", "Backlogs",
    ]


def _encode(profile: dict) -> pd.DataFrame:
    values = {
        **{name: float(profile[name]) for name in REQUIRED},
        "Gender": 0 if profile["Gender"] == "Female" else 1,
        **{name: profile[name] for name in ("Degree", "Branch")},
    }
    encoded = {}
    for column in _expected_columns():
        if column.startswith("Degree_"):
            encoded[column] = int(values["Degree"] == column.removeprefix("Degree_"))
        elif column.startswith("Branch_"):
            encoded[column] = int(values["Branch"] == column.removeprefix("Branch_"))
        else:
            encoded[column] = values[column]
    return pd.DataFrame([encoded], columns=_expected_columns())


def _validate(payload) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Send the student profile as a JSON object.")
    missing = (set(REQUIRED) | {"Gender", "Degree", "Branch"}) - payload.keys()
    if missing:
        raise ValueError(f"Complete all model fields: {', '.join(sorted(missing))}.")
    if payload["Gender"] not in VALID_GENDERS:
        raise ValueError("Choose Female or Male to match the training data categories.")
    if payload["Degree"] not in VALID_DEGREES:
        raise ValueError("Choose a degree category present in the training data.")
    if payload["Branch"] not in VALID_BRANCHES:
        raise ValueError("Choose a branch category present in the training data.")
    cleaned = dict(payload)
    for name, (minimum, maximum) in REQUIRED.items():
        try:
            value = float(payload[name])
        except (TypeError, ValueError):
            raise ValueError(f"{name.replace('_', ' ')} must be numeric.") from None
        if not np.isfinite(value) or value < minimum or value > maximum:
            raise ValueError(f"{name.replace('_', ' ')} must be between {minimum} and {maximum}.")
        if name not in {"CGPA"} and not value.is_integer():
            raise ValueError(f"{name.replace('_', ' ')} must be a whole number.")
        cleaned[name] = value
    cleaned["Degree"] = payload["Degree"]
    cleaned["Branch"] = payload["Branch"]
    cleaned["Gender"] = payload["Gender"]
    return cleaned


def _probability(profile: dict) -> float:
    row = _encode(profile)
    probabilities = model.predict_proba(row)[0]
    classes = list(model.classes_)
    placed_index = classes.index(1) if 1 in classes else classes.index("Placed")
    return float(probabilities[placed_index])


def _get_explainer():
    global explainer
    if explainer is None:
        with explainer_lock:
            if explainer is None:
                import shap
                explainer = shap.TreeExplainer(model)
    return explainer


def _explain(profile: dict) -> tuple[float, list[dict]]:
    row = _encode(profile)
    explanation = _get_explainer()
    values = explanation.shap_values(row, check_additivity=False)
    if isinstance(values, list):
        shap_row = np.asarray(values[1 if len(values) > 1 else 0])[0]
    else:
        shap_array = np.asarray(values)
        if shap_array.ndim == 3:
            shap_row = shap_array[0, :, 1]
        else:
            shap_row = shap_array[0]
    base = np.asarray(explanation.expected_value).reshape(-1)
    base_value = float(base[1] if base.size > 1 else base[0])
    explanations_by_feature = {
        name: {"feature": name, "value": 0.0, "display_value": profile[name]}
        for name in [*REQUIRED, "Gender", "Degree", "Branch"]
    }
    for column, value in zip(row.columns, shap_row):
        raw_name = column
        if column.startswith("Degree_"):
            raw_name = "Degree"
            display_value = profile["Degree"]
        elif column.startswith("Branch_"):
            raw_name = "Branch"
            display_value = profile["Branch"]
        elif column == "Gender":
            display_value = profile["Gender"]
            raw_name = "Gender"
        else:
            display_value = profile[column]
        # Combine the one-hot contributions into the visible category's effect.
        explanations_by_feature[raw_name]["value"] += float(value)
        explanations_by_feature[raw_name]["display_value"] = display_value
    explanations = list(explanations_by_feature.values())
    explanations.sort(key=lambda item: abs(item["value"]), reverse=True)
    return base_value, explanations


def _counterfactuals(profile: dict, current_probability: float) -> list[dict]:
    target = min(current_probability + 0.05, 1.0)
    options = []
    for feature, (direction, step, bound) in RECOMMENDABLE.items():
        start = float(profile[feature])
        if (direction == "increase" and start >= bound) or (direction == "decrease" and start <= bound):
            continue
        candidate = dict(profile)
        value = start
        best_probability = current_probability
        best_value = start
        target_value = None
        max_steps = int(round(abs(bound - start) / step))
        for count in range(1, max_steps + 1):
            value = start + (count * step if direction == "increase" else -count * step)
            value = round(value, 2) if feature == "CGPA" else int(value)
            candidate[feature] = value
            scenario_probability = _probability(candidate)
            if scenario_probability > best_probability:
                best_probability, best_value = scenario_probability, value
            if scenario_probability >= target:
                target_value = value
                break
        chosen = target_value if target_value is not None else best_value
        chosen_probability = _probability({**profile, feature: chosen})
        if chosen_probability > current_probability + 0.0005:
            options.append({
                "feature": feature,
                "direction": direction,
                "from": start,
                "to": chosen,
                "probability": chosen_probability,
                "delta": chosen_probability - current_probability,
                "reaches_target": target_value is not None,
            })
    options.sort(key=lambda item: (not item["reaches_target"], abs(item["to"] - item["from"]), -item["delta"]))
    return options[:3]


@app.get("/")
def home():
    return send_from_directory(APP_DIR, "index.html")


@app.get("/assets/shap_summary.png")
def shap_summary():
    return send_from_directory(PROJECT_DIR, "shap_summary.png")


@app.get("/api/health")
def health():
    return jsonify({"ready": model is not None, "error": model_error})


@app.post("/api/predict")
def predict():
    if model is None:
        return jsonify({"error": "The saved model could not be loaded. Check app dependencies and final_model.pkl."}), 503
    try:
        profile = _validate(request.get_json(silent=True))
        probability = _probability(profile)
        base_value, explanations = _explain(profile)
        recommendations = _counterfactuals(profile, probability)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": f"Model inference failed: {exc}"}), 500
    return jsonify({
        "prediction": "Placed" if probability >= 0.5 else "Not Placed",
        "probability": probability,
        "base_value": base_value,
        "explanations": explanations,
        "recommendations": recommendations,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
