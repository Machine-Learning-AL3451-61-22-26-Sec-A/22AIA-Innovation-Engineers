import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd

# Load and split data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Train model
model = GaussianNB()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

# Streamlit app
st.title("Naive Bayes Classifier ")
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# Input sliders
sepal_length = st.sidebar.slider("Sepal Length", float(iris.data[:, 0].min()), float(iris.data[:, 0].max()), float(iris.data[:, 0].mean()))
sepal_width = st.sidebar.slider("Sepal Width", float(iris.data[:, 1].min()), float(iris.data[:, 1].max()), float(iris.data[:, 1].mean()))
petal_length = st.sidebar.slider("Petal Length", float(iris.data[:, 2].min()), float(iris.data[:, 2].max()), float(iris.data[:, 2].mean()))
petal_width = st.sidebar.slider("Petal Width", float(iris.data[:, 3].min()), float(iris.data[:, 3].max()), float(iris.data[:, 3].mean()))

# Prediction
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)[0]

# Display results
st.write(f"Predicted Class: {iris.target_names[prediction]}")
st.write("Prediction Probabilities:")
st.write(pd.DataFrame(prediction_proba, index=iris.target_names, columns=["Probability"]))
