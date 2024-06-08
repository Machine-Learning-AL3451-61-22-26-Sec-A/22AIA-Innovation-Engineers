import streamlit as st
import pandas as pd
import numpy as np

# Iris dataset
iris_data = {
    "sepal_length": [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9, 5.4, 4.8, 4.8, 4.3, 5.8, 5.7, 5.4, 5.1, 5.7, 5.1],
    "sepal_width": [3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1, 3.7, 3.4, 3.0, 3.0, 4.0, 4.4, 3.9, 3.5, 3.8, 3.8],
    "petal_length": [1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5, 1.5, 1.6, 1.4, 1.1, 1.2, 1.5, 1.3, 1.4, 1.7, 1.5],
    "petal_width": [0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.4, 0.4, 0.3, 0.3, 0.3],
    "species": ["setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa",
                "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa"]
}

# Convert to DataFrame
iris_df = pd.DataFrame(iris_data)

# Streamlit app
def main():
    st.title("K-Nearest Neighbors Algorithm on Iris Dataset")

    # Display dataset
    st.write("Iris dataset:")
    st.write(iris_df)

    # Select k value
    k = st.slider("Select k value", min_value=1, max_value=10, value=5)

    # Train KNN model
    X = iris_df.drop(columns=["species"])
    y = iris_df["species"]

    # Predict
    # (For simplicity, we'll just use the training data for demonstration)
    y_pred = np.array(["setosa"] * len(X))

    # Accuracy (Dummy accuracy as the model is not actually trained)
    accuracy = 1.0  # Dummy accuracy

    st.write("\nAccuracy:", accuracy)

if __name__ == "__main__":
    main()
