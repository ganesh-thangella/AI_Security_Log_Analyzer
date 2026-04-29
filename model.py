import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import joblib

def preprocess_data(df):
    df = df.copy()

    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour

    # Encode categorical features
    le_protocol = LabelEncoder()
    le_status = LabelEncoder()

    df['protocol'] = le_protocol.fit_transform(df['protocol'])
    df['status'] = le_status.fit_transform(df['status'])

    features = df[['port', 'bytes', 'protocol', 'status', 'hour']]

    return features

def train_model(df):
    features = preprocess_data(df)

    model = IsolationForest(
        n_estimators=100,
        contamination=0.2,
        random_state=42
    )

    model.fit(features)
    joblib.dump(model, "anomaly_model.pkl")

    return model

def load_model():
    return joblib.load("anomaly_model.pkl")

def predict(model, df):
    features = preprocess_data(df)
    predictions = model.predict(features)
    scores = model.decision_function(features)

    df['anomaly'] = predictions
    df['risk_score'] = -scores

    return df