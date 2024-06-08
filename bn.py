import streamlit as st
import pandas as pd
from pomegranate import BayesianNetwork

# Sample data
data = pd.DataFrame({
    'A': [0, 0, 1, 1],
    'B': [0, 1, 0, 1],
    'C': [0, 1, 1, 1]
})

# Define Bayesian Network structure
model = BayesianNetwork.from_samples(data, algorithm='exact')

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

