# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Function to generate synthetic dataset (replace with your own data loading/preprocessing logic)
def generate_synthetic_data():
    # Generate synthetic dataset for demonstration
    data = {
        "feature1": [1.2, 2.5, 3.1, 4.6, 5.2],
        "feature2": [0.5, 1.2, 2.6, 3.5, 4.1],
        "target_column": [0, 1, 0, 1, 1]  # Binary target for classification
    }
    return pd.DataFrame(data)

# Function to train the Naive Bayes classifier
def train_model(X_train, y_train):
    # Initialize the Naive Bayes classifier
    model = GaussianNB()
    # Train the model
    model.fit(X_train, y_train)
    return model

# Function to evaluate the trained model
def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # Generate classification report
    report = classification_report(y_test, y_pred)
    return accuracy, report

# Streamlit app
def main():
    # Title of the web app
    st.title("Naive Bayes Classifier")

    # Generate synthetic data
    data = generate_synthetic_data()

    # Display the dataset
    st.subheader("Dataset")
    st.write(data)

    # Sidebar for model training
    st.sidebar.subheader("Model Training")
    # Split data into features and target variable
    X = data.drop(columns=["target_column"])
    y = data["target_column"]
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train the model
    model = train_model(X_train, y_train)
    # Evaluate the model
    accuracy, report = evaluate_model(model, X_test, y_test)
    st.sidebar.write("Accuracy:", accuracy)
    st.sidebar.write("Classification Report:")
    st.sidebar.write(report)

    # Prediction section
    st.subheader("Make Predictions")
    # User input for prediction
    input_data = {}
    # Example input fields, replace them with your own
    input_data["feature1"] = st.number_input("Feature 1", value=0.0)
    input_data["feature2"] = st.number_input("Feature 2", value=0.0)
    # Make prediction
    if st.button("Predict"):
        # Prepare input data for prediction
        input_df = pd.DataFrame([input_data])
        # Perform prediction
        prediction = model.predict(input_df)
        st.write("Predicted Class:", prediction[0])

# Run the app
if __name__ == "__main__":
    main()
