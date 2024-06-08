import streamlit as st
import pandas as pd

# Import the create_bayesian_network function from the separate module
from bayesian_network import create_bayesian_network

# Sample data
data = pd.DataFrame({
    'A': [0, 0, 1, 1],
    'B': [0, 1, 0, 1],
    'C': [0, 1, 1, 1]
})

# Create Bayesian Network
model = create_bayesian_network(data)

# Streamlit app
def main():
    st.title("Bayesian Network Visualization")

    st.write("### Sample Data")
    st.write(data)

    st.write("### Bayesian Network Structure")
    st.graphviz_chart(model.to_dot())

    st.write("### Conditional Probability Tables (CPTs)")
    for node in model.states:
        st.write(f"#### {node.name}")
        st.write(pd.DataFrame(node.distribution.parameters[0]))

if __name__ == "__main__":
    main()

