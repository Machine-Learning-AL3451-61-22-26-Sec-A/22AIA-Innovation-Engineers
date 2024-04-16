import streamlit as st

def is_consistent(hypothesis, instance):
    """
    Check if hypothesis is consistent with instance.
    """
    for i in range(len(hypothesis)):
        if hypothesis[i] != '?' and hypothesis[i] != instance[i]:
            return False
    return True

def generalize(hypothesis, instance):
    """
    Generalize hypothesis based on instance.
    """
    new_hypothesis = list(hypothesis)
    for i in range(len(hypothesis)):
        if hypothesis[i] == '?':
            new_hypothesis[i] = instance[i]
        elif hypothesis[i] != instance[i]:
            new_hypothesis[i] = '?'
    return tuple(new_hypothesis)

def specialize(hypotheses, instance, target):
    """
    Specialize the given hypothesis based on instance and target class.
    """
    new_hypotheses = []
    for hypothesis in hypotheses:
        if not is_consistent(hypothesis, instance):
            continue
        for i in range(len(hypothesis)):
            if hypothesis[i] == '?' and instance[i] != target[i]:
                new_hypothesis = list(hypothesis)
                new_hypothesis[i] = instance[i]
                new_hypotheses.append(tuple(new_hypothesis))
    return new_hypotheses

def candidate_elimination(instances):
    """
    Implement the Candidate Elimination Algorithm.
    """
    n = len(instances[0]) - 1  # Number of features
    G = [('?') * n]  # General boundary
    S = [None]  # Specific boundary
    
    for instance in instances:
        x, y = instance[:-1], instance[-1]
        if y == 'Yes':  # Positive example
            S = [s for s in S if is_consistent(s, x)]
            G = generalize(G, x)
            G = [g for g in G if any(is_consistent(g, s) for s in S)]
        else:  # Negative example
            S_prime = S[:]
            for s in S_prime:
                if is_consistent(s, x):
                    S.remove(s)
                    S += specialize(G, x, instance)
            G = [g for g in G if not any(is_consistent(g, s) for s in S)]
    return S, G

# Streamlit interface
def main():
    st.title("Candidate Elimination Algorithm")
    st.write("Enter instances as comma-separated values. Use 'Yes' for positive examples and 'No' for negative examples.")

    instances_input = st.text_area("Enter instances:", "Sunny,Hot,High,Weak,No\nSunny,Hot,High,Strong,No\nOvercast,Hot,High,Weak,Yes\nRain,Mild,High,Weak,Yes\nRain,Cool,Normal,Weak,Yes\nRain,Cool,Normal,Strong,No\nOvercast,Cool,Normal,Strong,Yes\nSunny,Mild,High,Weak,No\nSunny,Cool,Normal,Weak,Yes\nRain,Mild,Normal,Weak,Yes\nSunny,Mild,Normal,Strong,Yes\nOvercast,Mild,High,Strong,Yes\nOvercast,Hot,Normal,Weak,Yes\nRain,Mild,High,Strong,No")
    instances = [tuple(instance.split(',')) for instance in instances_input.split('\n') if instance.strip()]

    if st.button("Run Candidate Elimination Algorithm"):
        S, G = candidate_elimination(instances)
        st.write("Specific Boundary (S):", S)
        st.write("General Boundary (G):", G)

if __name__ == "__main__":
    main()
