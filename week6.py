import streamlit as st
import pandas as pd

# Sample data
data = pd.DataFrame({
    'A': [0, 0, 1, 1],
    'B': [0, 1, 0, 1],
    'C': [0, 1, 1, 1]
})

# Manually define Conditional Probability Tables (CPTs)
# These are simple example CPTs for illustration purposes
cpt_A = pd.DataFrame({'P(A=0)': [0.5], 'P(A=1)': [0.5]})
cpt_B_given_A = pd.DataFrame({'A': [0, 0, 1, 1], 'P(B=0|A)': [0.8, 0.2, 0.3, 0.7], 'P(B=1|A)': [0.2, 0.8, 0.7, 0.3]})
cpt_C_given_B = pd.DataFrame({'B': [0, 0, 1, 1], 'P(C=0|B)': [0.9, 0.1, 0.4, 0.6], 'P(C=1|B)': [0.1, 0.9, 0.6, 0.4]})

# Streamlit app
def main():
    st.title("Simple Bayesian Network Visualization")

    st.write("### Sample Data")
    st.write(data)

    st.write("### Conditional Probability Tables (CPTs)")

    st.write("#### P(A)")
    st.write(cpt_A)

    st.write("#### P(B|A)")
    st.write(cpt_B_given_A)

    st.write("#### P(C|B)")
    st.write(cpt_C_given_B)

    st.write("### Network Structure")
    st.graphviz_chart('''
        digraph {
            A -> B
            B -> C
        }
    ''')

if __name__ == "__main__":
    main()
