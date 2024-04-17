import streamlit as st

def is_consistent(h, e):
    for i in range(len(h)):
        if h[i] != '?' and h[i] != e[i]:
            return False
    return True

def generalize_S(h, e):
    h_new = list(h)
    for i in range(len(h)):
        if h[i] == '?':
            h_new[i] = e[i]
        elif h[i] != e[i]:
            h_new[i] = '?'
    return tuple(h_new)

def specialize_G(D, G, e):
    S = [g for g in G if is_consistent(g, e)]
    for s in S[:]:
        if not any(is_consistent(s, d) and not all(d[i] == s[i] or s[i] == '?' for i in range(len(s))) for d in D):
            S.remove(s)
    return S

def candidate_elimination(D):
    G = [tuple(['?' for _ in range(len(D[0])-1)])]  # Initialize G with most specific hypothesis
    S = [tuple(['0' for _ in range(len(D[0])-1)])]  # Initialize S with most general hypothesis
    for d in D:
        x, y = d[:-1], d[-1]
        if y == 'Y':  # Positive example
            G = [h for h in G if is_consistent(h, x)]
            S_new = []
            for s in S:
                if is_consistent(s, x):
                    S_new.append(s)
                else:
                    S_new.extend([h for h in generalize_S(s, x) if h not in S_new])
            S = S_new
            G = specialize_G(D, G, x)
        else:  # Negative example
            S = [s for s in S if not is_consistent(s, x)]
            G_new = []
            for g in G:
                if not is_consistent(g, x):
                    G_new.append(g)
                else:
                    G_new.extend([h for h in generalize_S(g, x) if h not in G_new])
            G = G_new
            S = specialize_G(D, S, x)
        st.write("G:", G)
        st.write("S:", S)

def main():
    st.title("Candidate Elimination Algorithm")

    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data:")
        st.write(data)

        st.header("Candidate Elimination Steps:")
        candidate_elimination(data.values)

if __name__ == "__main__":
    main()
