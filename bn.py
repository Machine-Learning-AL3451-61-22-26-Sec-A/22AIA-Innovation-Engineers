import streamlit as st
from bayesian_network import create_bayesian_network, perform_inference

st.title("Bayesian Network Inference")

# Create the Bayesian Network
model = create_bayesian_network()

st.sidebar.header("Input Evidence")
a_value = st.sidebar.selectbox("A", [0, 1], format_func=lambda x: f"A = {x}")
b_value = st.sidebar.selectbox("B", [0, 1], format_func=lambda x: f"B = {x}")

evidence = {'A': a_value, 'B': b_value}

if st.sidebar.button("Infer"):
    result = perform_inference(model, evidence)
    st.write(f"With evidence A={a_value} and B={b_value}, the most likely state of C is {result['C']}.")
else:
    st.write("Set evidence in the sidebar and click 'Infer' to perform inference.")

st.write("## Bayesian Network Structure")
st.graphviz_chart("""
digraph {
    A -> C;
    B -> C;
}
""")
