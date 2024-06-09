import streamlit as st
import pandas as pd

# Function to check if a hypothesis is consistent with an instance
def is_consistent(h, instance):
    return all(hv == '?' or hv == iv for hv, iv in zip(h, instance))

# Candidate Elimination Algorithm
def candidate_elimination(examples):
    specific_h = ['0'] * (len(examples.columns) - 1)
    general_h = [['?' for _ in specific_h]]
    
    for _, row in examples.iterrows():
        instance, label = row[:-1], row[-1]
        if label == 'Yes':
            specific_h = [iv if sh == '0' else '?' if sh != iv else sh for sh, iv in zip(specific_h, instance)]
            general_h = [g for g in general_h if is_consistent(g, instance)]
        else:
            new_general_h = []
            for g in general_h:
                new_general_h.extend([g[:i] + [specific_h[i]] + g[i+1:] for i in range(len(g)) if g[i] == '?'])
            general_h = [g for g in new_general_h if any(is_consistent(g, x[:-1]) for _, x in examples[examples['EnjoySport'] == 'Yes'].iterrows())]
    
    return specific_h, general_h

# Streamlit app
def main():
    st.write("22AIA-INNOVATIVE ENGINEERS")
    st.title("Candidate Elimination Algorithm")

    # Sample data
    data = {'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny'],
            'AirTemp': ['Warm', 'Warm', 'Cold', 'Warm'],
            'Humidity': ['Normal', 'High', 'High', 'High'],
            'Wind': ['Strong', 'Strong', 'Strong', 'Strong'],
            'Water': ['Warm', 'Warm', 'Warm', 'Cool'],
            'Forecast': ['Same', 'Same', 'Change', 'Change'],
            'EnjoySport': ['Yes', 'Yes', 'No', 'Yes']}
    examples = pd.DataFrame(data)

    st.write("### Training Examples")
    st.write(examples)

    specific_h, general_h = candidate_elimination(examples)

    st.write("### Specific Hypothesis")
    st.write(specific_h)

    st.write("### General Hypotheses")
    st.write(general_h)

if __name__ == "__main__":
    main()
