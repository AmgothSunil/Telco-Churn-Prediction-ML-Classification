import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# Read data
def read_data(path):
    df = pd.read_csv(path)
    return df

# EDA / Preprocessing
def eda(df):
    # Drop irrelevant column
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(" ", np.nan))
    df.fillna({'TotalCharges': 0.0}, inplace=True)

    return df

# Encode target variable
def target_labelling(df, target='Churn'):
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target])
    return df

# Train model with pipeline
def train_model(df, target='Churn'):
    # Separate features and target
    X = df.drop(target, axis=1)
    y = df[target]

    # Identify categorical and numerical columns
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Numerical pipeline
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Column transformer
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_columns),
        ('cat', cat_pipeline, categorical_columns)
    ])

    # Full pipeline with SMOTE and classifier
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', LogisticRegression(
            penalty='l1', C=0.5, solver='saga', max_iter=1000, random_state=42
        ))
    ])

    # Train the pipeline
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

    # Save model
    joblib.dump(pipeline, "model.joblib")
    print("Pipeline saved as model.joblib")

    return pipeline

# Example usage
if __name__ == "__main__":
    df = read_data("D:/AI Projects/tele_churn/frontend/dataset/tele_churn_data.csv")
    df = eda(df)
    df = target_labelling(df, target='Churn')
    model_pipeline = train_model(df, target='Churn')
