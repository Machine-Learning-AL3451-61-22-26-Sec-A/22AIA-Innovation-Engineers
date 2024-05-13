import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split  # Add this line
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Generate synthetic dataset
data = pd.DataFrame({
    "feature1": [1.2, 2.5, 3.1, 4.6, 5.2],
    "feature2": [0.5, 1.2, 2.6, 3.5, 4.1],
    "target_column": [0, 1, 0, 1, 1]
})

# Train the Naive Bayes classifier
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=["target_column"]), data["target_column"], test_size=0.2, random_state=42)
model = GaussianNB().fit(X_train, y_train)

# Streamlit app
st.title("Naive Bayes Classifier")

# Display the dataset
st.subheader("Dataset")
st.write(data)

# Model evaluation
st.sidebar.subheader("Model Evaluation")
accuracy = accuracy_score(y_test, model.predict(X_test))
report = classification_report(y_test, model.predict(X_test))
st.sidebar.write("Accuracy:", accuracy)
st.sidebar.write("Classification Report:")
st.sidebar.write(report)

# Prediction section
st.subheader("Make Predictions")
input_data = {
    "feature1": st.number_input("Feature 1", value=0.0),
    "feature2": st.number_input("Feature 2", value=0.0)
}
if st.button("Predict"):
    prediction = model.predict(pd.DataFrame([input_data]))
    st.write("Predicted Class:", prediction[0])
