import streamlit as st

def candidate_elimination(concepts, targets):
    specific_h = concepts[0].copy()
    general_h = [['?' for _ in range(len(specific_h))] for _ in range(len(specific_h))]

    for i, (x, y) in enumerate(zip(concepts, targets)):
        if y == 'Yes':
            for j in range(len(specific_h)):
                if x[j] != specific_h[j]:
                    specific_h[j] = '?'
                    general_h[j][j] = '?'
        elif y == 'No':
            for j in range(len(specific_h)):
                if x[j] != specific_h[j]:
                    general_h[j][j] = specific_h[j]
                else:
                    general_h[j][j] = '?'
    
    return specific_h, general_h

def main():
    st.title("Candidate Elimination Algorithm")

    num_attributes = st.number_input("Enter the number of attributes:", min_value=1, step=1, value=1)
    num_examples = st.number_input("Enter the number of examples:", min_value=1, step=1, value=1)

    st.write("Enter attributes and target values (Yes/No) for each example:")
    concepts = []
    targets = []
    for i in range(num_examples):
        concept = []
        for j in range(num_attributes):
            concept.append(st.text_input(f"Attribute {j+1} for Example {i+1}:"))
        target = st.selectbox(f"Target value for Example {i+1}:", options=['Yes', 'No'])
        concepts.append(concept)
        targets.append(target)

    if st.button("Run Candidate Elimination Algorithm"):
        specific, general = candidate_elimination(concepts, targets)
        st.write("Specific Hypothesis:", specific)
        st.write("General Hypothesis:", general)

if _name_ == "__main__ ":
    main()
