import streamlit as st
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def backpropagation(X, y, num_hidden_layers, num_neurons, learning_rate, epochs):
    # Initialize weights
    input_layer_size = X.shape[1]
    output_layer_size = y.shape[1]
    weights = []
    for i in range(num_hidden_layers + 1):
        if i == 0:
            weights.append(2 * np.random.random((input_layer_size, num_neurons)) - 1)
        elif i == num_hidden_layers:
            weights.append(2 * np.random.random((num_neurons, output_layer_size)) - 1)
        else:
            weights.append(2 * np.random.random((num_neurons, num_neurons)) - 1)

    # Training
    for _ in range(epochs):
        layers = [X]
        layer_outputs = [X]
        # Forward pass
        for i in range(num_hidden_layers + 1):
            layer_input = np.dot(layers[i], weights[i])
            layer_output = sigmoid(layer_input)
            layers.append(layer_input)
            layer_outputs.append(layer_output)
        
        # Backpropagation
        errors = [y - layer_outputs[-1]]
        for i in range(num_hidden_layers, 0, -1):
            error = errors[-1].dot(weights[i].T)
            errors.append(error * sigmoid_derivative(layer_outputs[i]))

        # Update weights
        for i in range(num_hidden_layers + 1):
            weights[i] += learning_rate * layers[i].T.dot(errors[num_hidden_layers - i])

    return weights

def main():
    st.title("Backpropagation Algorithm")

    # User inputs
    num_hidden_layers = st.number_input("Number of hidden layers:", min_value=1, step=1, value=1)
    num_neurons = st.number_input("Number of neurons per hidden layer:", min_value=1, step=1, value=4)
    learning_rate = st.number_input("Learning rate:", min_value=0.01, max_value=1.0, step=0.01, value=0.1)
    epochs = st.number_input("Number of epochs:", min_value=1, step=1, value=1000)

    # Training data
    st.subheader("Training Data")
    num_samples = st.number_input("Number of training samples:", min_value=1, step=1, value=4)
    X = []
    y = []
    for i in range(num_samples):
        st.write(f"Sample {i + 1}:")
        x_input = st.text_input("Input (comma-separated values):")
        x_values = list(map(float, x_input.split(',')))
        X.append(x_values)
        y_input = st.selectbox("Output:", options=[0, 1])
        y.append([y_input])

    X = np.array(X)
    y = np.array(y)

    if st.button("Train"):
        weights = backpropagation(X, y, num_hidden_layers, num_neurons, learning_rate, epochs)
        st.success("Training completed!")

        # Display trained weights
        st.subheader("Trained Weights")
        for i, w in enumerate(weights):
            st.write(f"Layer {i} weights:")
            st.write(w)

if __name__ == "__main__":
    main()
