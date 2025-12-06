import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Import the cleaner we just wrote
from src.preprocess import clean_and_prepare


def train():
    # Setup folders
    if not os.path.exists('model'):
        os.makedirs('model')

    print("--- Step 1: Loading Data ---")
    df = pd.read_csv('data/loan.csv')

    print("--- Step 2: Cleaning Data ---")
    # Clean the inputs and save the translation rules (encoders)
    X_clean, encoders = clean_and_prepare(df, train=True)

    # Fix the Target (The Answer Key): Convert Y/N to 1/0
    target_le = LabelEncoder()
    X_clean['Loan_Status'] = target_le.fit_transform(X_clean['Loan_Status'])

    # Separate Inputs (X) and Answers (y)
    y = X_clean['Loan_Status']
    X = X_clean.drop('Loan_Status', axis=1)

    # Split: 80% for studying, 20% for the final exam
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("--- Step 3: Training the Brain ---")
    # We use Logistic Regression (Simple and Effective)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Test it
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"✅ Model Accuracy: {accuracy:.2%}")

    print("--- Step 4: Saving Files ---")
    joblib.dump(model, 'model/loan_model.pkl')
    joblib.dump(encoders, 'model/encoders.pkl')
    joblib.dump(list(X.columns), 'model/feature_names.pkl')
    print("🎉 Done! Ready for the GUI.")


if __name__ == "__main__":
    train()