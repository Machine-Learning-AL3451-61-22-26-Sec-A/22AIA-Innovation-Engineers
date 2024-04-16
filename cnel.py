import numpy as np

class CandidateElimination:
    def __init__(self, num_features):
        # Initialize the general and specific hypotheses
        self.G = [['?' for _ in range(num_features)]]
        self.S = [['0' for _ in range(num_features)]]

    def fit(self, X, y):
        for i in range(len(X)):
            if y[i] == 'Y':
                # Remove inconsistent hypotheses from G
                self.G = [g for g in self.G if self.is_consistent(g, X[i])]
                # Generalize S
                self.S = self.generalize_S(X[i])
            else:
                # Remove inconsistent hypotheses from S
                self.S = [s for s in self.S if not self.is_consistent(s, X[i])]
                # Specialize G
                self.G = self.specialize_G(X[i])

    def is_consistent(self, hypothesis, instance):
        for i in range(len(hypothesis)):
            if hypothesis[i] != '?' and hypothesis[i] != instance[i]:
                return False
        return True

    def generalize_S(self, instance):
        S_new = []
        for s in self.S:
            for i in range(len(s)):
                if s[i] == '0':
                    s_copy = s.copy()
                    s_copy[i] = instance[i]
                    if self.is_consistent(s_copy, instance):
                        S_new.append(s_copy)
        return S_new

    def specialize_G(self, instance):
        G_new = []
        for g in self.G:
            for i in range(len(g)):
                if g[i] == '?':
                    for val in ['0', '1']:
                        if instance[i] != val:
                            g_copy = g.copy()
                            g_copy[i] = val
                            consistent = True
                            for s in self.S:
                                if not self.is_consistent(s, g_copy):
                                    consistent = False
                                    break
                            if consistent:
                                G_new.append(g_copy)
        return G_new

    def get_hypotheses(self):
        return self.G, self.S

# Example usage:
X = np.array([
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change']
])

y = np.array(['Y', 'Y', 'N', 'Y'])

ce = CandidateElimination(num_features=X.shape[1])
ce.fit(X, y)
G, S = ce.get_hypotheses()

print("Final General Hypotheses:")
for g in G:
    print(g)

print("\nFinal Specific Hypotheses:")
for s in S:
    print(s)
