import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def clean_and_prepare(df, train=True, encoders=None):
    """
    This function takes messy data and turns it into clean numbers.

    Args:
        df: The data (one row or the whole dataset).
        train: True if we are learning, False if we are predicting.
        encoders: Saved rules for translation (used when train=False).
    """
    df = df.copy()

    # 1. Remove ID column (Computers don't need names/IDs to predict)
    if 'Loan_ID' in df.columns:
        df = df.drop(columns=['Loan_ID'])

    # 2. Identify which columns are Words and which are Numbers
    words_cols = ['Gender', 'Married', 'Dependents', 'Education',
                  'Self_Employed', 'Property_Area']

    nums_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                 'Loan_Amount_Term', 'Credit_History']

    # 3. Fill Missing Values (The "Imputation" Step)
    # If words are missing, fill with the most common one.
    imputer_words = SimpleImputer(strategy='most_frequent')
    df[words_cols] = imputer_words.fit_transform(df[words_cols])

    # If numbers are missing, fill with the average.
    imputer_nums = SimpleImputer(strategy='mean')
    df[nums_cols] = imputer_nums.fit_transform(df[nums_cols])

    # 4. Convert Words to Numbers & Scale Data
    processed_encoders = {} if train else encoders

    for col in words_cols:
        if train:
            # Create a new translator (e.g., Male -> 1, Female -> 0)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            processed_encoders[col] = le
        else:
            # Use the existing translator
            le = processed_encoders[col]
            # Handle tricky case: if we see a new word we don't know, use the first known one
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])

    # 5. Scale Numbers (So Income=5000 doesn't overpower Credit=1)
    if train:
        scaler = StandardScaler()
        df[nums_cols] = scaler.fit_transform(df[nums_cols])
        processed_encoders['scaler'] = scaler
    else:
        scaler = processed_encoders['scaler']
        df[nums_cols] = scaler.transform(df[nums_cols])

    return df, processed_encoders