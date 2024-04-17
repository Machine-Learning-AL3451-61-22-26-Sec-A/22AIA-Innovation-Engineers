import streamlit as st

def candidate_elimination(examples):
    # Initialize the hypothesis space
    specific_h = examples[0][:-1]
    general_h = [['?' for _ in range(len(specific_h))] for _ in range(len(specific_h))]

    for example in examples:
        x, y = example[:-1], example[-1]

        if y == 'Y':
            for i in range(len(specific_h)):
                if x[i] != specific_h[i]:
                    specific_h[i] = '?'
                    general_h[i][i] = '?'
        else:
            for i in range(len(specific_h)):
                if x[i] != specific_h[i]:
                    general_h[i][i] = specific_h[i]
                else:
                    general_h[i][i] = '?'

        general_h = [h for h in general_h if h != ['?' for _ in range(len(specific_h))]]

    return specific_h, general_h

def main():
    st.title("Candidate Elimination Algorithm")

    # Sample dataset
    sample_dataset = [
        ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Y'],
        ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Y'],
        ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'N'],
        ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Y']
    ]

    st.subheader("Sample Dataset")
    for row in sample_dataset:
        st.write(row)

    specific_h, general_h = candidate_elimination(sample_dataset)

    # Output results
    st.subheader("Output Results")
    
    st.write("**Specific Hypothesis:**")
    st.write(specific_h)
    st.write("**General Hypotheses:**")
    st.write(general_h)
    
    # Print results
    st.text("\n\nOutput printed in console:")
    print("Specific Hypothesis:")
    print(specific_h)
    print("\nGeneral Hypotheses:")
    for h in general_h:
        print(h)

if __name__ == "__main__":
    main()
