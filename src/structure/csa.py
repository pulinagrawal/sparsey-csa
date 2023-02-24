import numpy as np
import math

class Macrocolumn:

    def __init__(self):
        input_size = 512
        sparsity = 0.03
        code_size = 2000
        self.n_pyramidal_per_minicol = int(code_size*sparsity)
        self.n_minicol = int(code_size/self.n_pyramidal_per_minicol)

        self.weights = np.zeros((self.n_minicol, self.n_pyramidal_per_minicol, input_size))

        self.state = np.zeros((self.n_minicol, self.n_pyramidal_per_minicol))
        self.pyramidal_indices = None


    def normalized_overlap(self, input):
        input_activity = sum(input)
        u = np.sum(self.weights * input, axis=2)
        V = u/input_activity
        return V

    def learn_on(self, pyramidal_indices, input):
        for minicol, pyramidal in enumerate(pyramidal_indices):
            for i, weight in enumerate(self.weights[minicol][pyramidal]):
                if input[i]==1 and weight!=1:
                    self.weights[minicol][pyramidal][i]=1

    def run(self, input, learn=True):
        #check input
        V = self.normalized_overlap(input)
        G = np.mean(np.max(V, axis=1))

        a=10000
        b=7
        n=a*(G**b)

        lmbda = 26
        sm_fi = -13

        fi = (n/(1+math.e**(-(lmbda*V+sm_fi))))+1

        rho = fi/ np.sum(fi, axis=1)[:,None]

        self.state.fill(0)

        self.pyramidal_indices = [ np.random.choice(range(len(minicol)), p=minicol) for minicol in rho ]

        for minicol, pyramidal in enumerate(self.pyramidal_indices):
            self.state[minicol][pyramidal]=1

        if learn:
            self.learn_on(self.pyramidal_indices, input)

        return self.state

