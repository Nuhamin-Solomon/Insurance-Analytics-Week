import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    mean_squared_error, r2_score, 
    roc_auc_score, classification_report
)

# --- 1. Data Preparation and Preprocessing ---

def create_preprocessor(numerical_features, categorical_features):
    """
    Creates a column transformer pipeline for scaling numerical features 
    and one-hot encoding categorical features.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor

def prepare_data_for_modeling(df):
    """
    Separates the data into two sets: one for Claim Frequency (Classification)
     and one for Claim Severity (Regression).
     
    NOTE: You must adjust the features based on your final EDA and business understanding.
    """
    # Define features based on the data structure (using placeholders from the data description)
    common_features = [
        'Cylinders', 'Cubiccapacity', 'Kilowatts', 'CustomValueEstimate', 
        'SumInsured', 'TotalPremium', 'Province', 'Gender', 'VehicleType', 
        'Make', 'Bodytype', 'MaritalStatus', 'AgeOfDriver' 
    ]
    
    numerical_features = ['Cylinders', 'Cubiccapacity', 'Kilowatts', 'CustomValueEstimate', 'SumInsured', 'TotalPremium', 'AgeOfDriver']
    categorical_features = ['Province', 'Gender', 'VehicleType', 'Make', 'Bodytype', 'MaritalStatus']
    
    # --- 1. Claim Frequency Data (Classification Target: HasClaim) ---
    X_freq = df[common_features].copy()
    y_freq = df['HasClaim']
    
    X_train_freq, X_test_freq, y_train_freq, y_test_freq = train_test_split(
        X_freq, y_freq, test_size=0.2, random_state=42, stratify=y_freq
    )
    
    # --- 2. Claim Severity Data (Regression Target: TotalClaims | Filtered for claims > 0) ---
    severity_df = df[df['HasClaim'] == 1].dropna(subset=['TotalClaims']).copy()
    X_sev = severity_df[common_features].copy()
    y_sev = severity_df['TotalClaims']
    
    X_train_sev, X_test_sev, y_train_sev, y_test_sev = train_test_split(
        X_sev, y_sev, test_size=0.2, random_state=42
    )

    return (numerical_features, categorical_features, 
            X_train_freq, X_test_freq, y_train_freq, y_test_freq, 
            X_train_sev, X_test_sev, y_train_sev, y_test_sev)


# --- 2. Model Training and Evaluation Functions ---

def train_and_evaluate_classification(preprocessor, X_train, X_test, y_train, y_test, models):
    """Trains and evaluates classification models for Claim Frequency (Probability)."""
    results = {}
    
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', model)])
        
        print(f"\n--- Training {name} for Claim Frequency ---")
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        # Use predict_proba for AUC score
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Evaluation
        auc_score = roc_auc_score(y_test, y_proba)
        
        print(f"AUC Score: {auc_score:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        results[name] = {'model': pipeline, 'AUC': auc_score}
        
    return results

def train_and_evaluate_regression(preprocessor, X_train, X_test, y_train, y_test, models):
    """Trains and evaluates regression models for Claim Severity (Amount)."""
    results = {}

    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', model)])
        
        print(f"\n--- Training {name} for Claim Severity ---")
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        
        # Evaluation 
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"RMSE: {rmse:.2f}")
        print(f"R-squared: {r2:.4f}")
        
        results[name] = {'model': pipeline, 'RMSE': rmse, 'R2': r2}
        
    return results

# --- 3. Model Interpretation Helper ---

def get_feature_names(preprocessor, numerical_features, categorical_features):
    """Generates the full list of feature names after preprocessing."""
    # Get the feature names from the OneHotEncoder step
    ohe_features = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
    
    # Combine numerical features and one-hot encoded features
    all_features = numerical_features + ohe_features
    
    return all_features