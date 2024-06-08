import streamlit as st
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import pandas as pd

# Define Bayesian Network structure and CPDs
model = BayesianNetwork([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')])
model.add_cpds(
    TabularCPD('A', 2, [[0.6], [0.4]]),
    TabularCPD('B', 2, [[0.7, 0.2], [0.3, 0.8]], evidence=['A'], evidence_card=[2]),
    TabularCPD('C', 2, [[0.8, 0.5], [0.2, 0.5]], evidence=['A'], evidence_card=[2]),
    TabularCPD('D', 2, [[0.9, 0.6, 0.7, 0.1], [0.1, 0.4, 0.3, 0.9]], evidence=['B', 'C'], evidence_card=[2, 2])
)
assert model.check_model()
inference = VariableElimination(model)

# Streamlit app
st.title("Simple Bayesian Network Inference")

# Sidebar for user inputs
st.sidebar.title("Input Evidence")
evidence = {var: val for var, val in zip(['A', 'B', 'C', 'D'], [
    st.sidebar.selectbox("A", [None, 0, 1]),
    st.sidebar.selectbox("B", [None, 0, 1]),
    st.sidebar.selectbox("C", [None, 0, 1]),
    st.sidebar.selectbox("D", [None, 0, 1])
]) if val is not None}

# Perform inference
if evidence:
    query = inference.map_query(variables=['A', 'B', 'C', 'D'], evidence=evidence)
    st.write("### Inference Results")
    st.write(pd.DataFrame(query.items(), columns=["Variable", "State"]))
else:
    st.write("### Set evidence in the sidebar to perform inference.")

# Display CPDs
st.write("### Conditional Probability Distributions (CPDs)")
for cpd in model.get_cpds():
    st.write(f"CPD of {cpd.variable}:")
    st.write(cpd)
