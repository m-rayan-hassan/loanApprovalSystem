import joblib
import pandas as pd
import numpy as np
from src.preprocess import clean_and_prepare


class LoanPredictor:
    def __init__(self):
        # Load the saved files
        self.model = joblib.load('model/loan_model.pkl')
        self.encoders = joblib.load('model/encoders.pkl')
        self.features = joblib.load('model/feature_names.pkl')

    def predict(self, user_input):
        # 1. Turn user input (dictionary) into a DataFrame
        df = pd.DataFrame([user_input])

        # 2. Clean it using the SAVED encoders (train=False)
        df_clean, _ = clean_and_prepare(df, train=False, encoders=self.encoders)

        # 3. Make Prediction
        # [0] means "Rejected", [1] means "Approved"
        prediction_idx = self.model.predict(df_clean)[0]

        # Get Probability (Confidence)
        probs = self.model.predict_proba(df_clean)[0]
        confidence = probs[prediction_idx]

        status = "Approved" if prediction_idx == 1 else "Rejected"

        # 4. Calculate Risk Level
        risk_score = probs[0]  # Probability of Rejection
        if risk_score < 0.30:
            risk = "Low"
        elif risk_score < 0.70:
            risk = "Medium"
        else:
            risk = "High"

        # 5. Get Top Factors (For the Graph)
        # We look at the "Weights" (coefficients) of the model
        weights = np.abs(self.model.coef_[0])
        sorted_idx = np.argsort(weights)[::-1]  # Sort biggest to smallest

        top_features = {}
        for i in range(3):  # Top 3
            idx = sorted_idx[i]
            name = self.features[idx]
            top_features[name] = weights[idx]

        return {
            "status": status,
            "confidence": f"{confidence:.0%}",
            "risk": risk,
            "risk_score": f"{risk_score:.0%}",
            "top_features": top_features
        }