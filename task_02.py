import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    jaccard_score,
    f1_score,
    log_loss,
)
import tensorflow as tf
import matplotlib.pyplot as plt

# Function for loading and preparing data with descriptive variable names
def load_and_preprocess_data(file_path, n_rows):
    data = pd.read_csv(file_path)
    data = data.drop(columns=["Unnamed: 0", "transaction_number", "address"])
    prepared_data = pd.get_dummies(data=data.head(n=n_rows))
    return prepared_data

# Function for training and evaluating models with streamlined structure
def evaluate_machine_learning_models(X_train, y_train, X_test, y_test):
    models = [
        LogisticRegression(solver="liblinear"),
        DecisionTreeClassifier(),
        KNeighborsClassifier(n_neighbors=4),
        SVC(),
        # Add more models as needed
    ]

    for model in models:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Calculate and print common evaluation metrics
        accuracy = accuracy_score(y_test, predictions)
        jaccard = jaccard_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        model_name = model.__class__.__name__
        print(f"{model_name} Accuracy: {accuracy:.4f}")
        print(f"{model_name} Jaccard Index: {jaccard:.4f}")
        print(f"{model_name} F1 Score: {f1:.4f}")

        # Calculate and print additional metrics specific to each model
        if model_name == "LogisticRegression":
            log_loss_value = log_loss(y_test, model.predict_proba(X_test))
            print(f"{model_name} Log Loss: {log_loss_value:.4f}")
        elif model_name in ["DecisionTreeClassifier", "KNeighborsClassifier", "SVC"]:
            # Additional metrics for these models can be added as needed
            pass
        elif isinstance(model, tf.keras.models.Sequential):
            # Add metrics from the `history` object obtained during model training
            pass

# Function for loading and preparing data with descriptive variable names
def load_and_preprocess_data(file_path, n_rows):
    data = pd.read_csv(file_path)
    print("Columns in the loaded data:", data.columns)  # Print columns for debugging
    data = data.drop(columns=["Unnamed: 0", "transaction_number", "address"], errors="ignore")
    print("Columns after dropping:", data.columns)  # Print columns after dropping for debugging
    prepared_data = pd.get_dummies(data=data.head(n=n_rows))
    return prepared_data

# Load and prepare the data
training_data = load_and_preprocess_data("archive/fraudTrain.csv", 20000)
X_train = training_data.drop(columns="is_fraud", axis=1)
y_train = training_data["is_fraud"]

# Verify if X_train and y_train are defined
print("X_train shape:", X_train.shape)  # Print the shape of X_train
print("y_train shape:", y_train.shape)  # Print the shape of y_train


test_data = load_and_preprocess_data("archive/fraudTest.csv", 5000)
X_test = test_data.drop(columns="is_fraud", axis=1)
y_test = test_data["is_fraud"]

# Align columns in X_train and X_test
X_train, X_test = X_train.align(X_test, join="outer", axis=1, fill_value=0)

# Evaluate the models
evaluate_machine_learning_models(X_train, y_train, X_test, y_test)
