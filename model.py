import joblib
import shap
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


def build_model():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )


def train_model(X, y):
    model = build_model()
    model.fit(X, y)
    return model


def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    return {
        "accuracy": accuracy_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_proba),
        "classification_report": classification_report(y, y_pred)
    }


def explain_predictions(model, X, feature_names, top_n=3):

    X_sample = X.sample(min(50, len(X)), random_state=42)

    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample, check_additivity=False)

    results = []

    for i in range(min(top_n, len(X_sample))):

        proba = model.predict_proba(X_sample.iloc[[i]])[0][1]
        prediction = "Theft" if proba > 0.3 else "Normal"

        shap_row = shap_values.values[i][:, 1]   

        explanation = pd.Series(
            shap_row,
            index=feature_names
        ).sort_values(ascending=False)

        results.append({
            "index": i,
            "prediction": prediction,
            "score": float(proba),
            "top_features": explanation.head(5).to_dict()
        })

    return results


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    model = joblib.load(path)
    return model, None