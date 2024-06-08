import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Generate a synthetic dataset
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = np.cos(X) + np.random.normal(0, 0.1, 100)
data = pd.DataFrame({'X': X, 'y': y})

# Locally Weighted Regression (LWR) function
def locally_weighted_regression(x, X, y, tau):
    weights = np.exp(-np.sum((X - x)**2, axis=1) / (2 * tau**2))
    W = np.diag(weights)
    X_ = np.vstack([np.ones(len(X)), X]).T
    theta = np.linalg.inv(X_.T @ W @ X_) @ X_.T @ W @ y
    return np.dot([1, x], theta)

# Streamlit app
def main():
    st.title("Locally Weighted Regression")

    # Display dataset
    st.write("Dataset:")
    st.write(data.head())

    # Select tau value
    tau = st.slider("Select tau value (bandwidth parameter)", min_value=0.1, max_value=1.0, value=0.5, step=0.01)

    # Fit LWR model and predict
    y_pred = np.array([locally_weighted_regression(x, X, y, tau) for x in X])
    predictions = pd.DataFrame({'X': X, 'Actual y': y, 'Predicted y': y_pred})

    # Plot results using Altair
    base = alt.Chart(predictions).encode(x='X')
    points = base.mark_circle(color='blue').encode(y='Actual y')
    line = base.mark_line(color='red').encode(y='Predicted y')

    st.altair_chart(points + line, use_container_width=True)

if __name__ == "__main__":
    main()

