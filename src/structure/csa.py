import numpy as np
import math

class Macrocolumn:

    def __init__(self):
        input_size = 512
        sparsity = 0.02
        code_size = 1024
        n_minicol = int(code_size/sparsity)
        n_pyramidal_per_minicol = int(code_size/n_minicol)

        self.weights = np.zeros((n_minicol, n_pyramidal_per_minicol, input_size))

        self.state = np.zeros((n_minicol, n_pyramidal_per_minicol))


    def normalized_overlap(self, weights, input):
        input_activity = sum(input)
        u = np.dot(weights, input)
        V = u/input_activity
        return V

    def learn_on(self, pyramidal_indices, weights, input):
        for minicol, pyramidal in enumerate(pyramidal_indices):
            for i, weight in enumerate(weights[minicol][pyramidal]):
                if input[i]==1 and weight!=1:
                    self.weights[minicol][pyramidal][i]=1

    def run(self, input, learn=True):
        #check input
        V = self.normalized_overlap(self.weights, input)
        G = np.mean(np.max(V, axis=1))

        a=4.6
        f=2.8
        n=math.e**(4.6*(G**f))-1

        lmbda = 28
        sm_fi = -5

        fi = (n/(1+math.e**(-(lmbda*V+sm_fi))))+1

        rho = fi/ np.sum(fi, axis=1)[:,None]

        self.state.fill(0)

        pyramidal_indices = [ np.random.choice(range(len(minicol)), p=minicol) for minicol in rho ]

        for minicol, pyramidal in enumerate(pyramidal_indices):
            self.state[minicol][pyramidal]=1

        if learn:
            self.learn_on(pyramidal_indices, self.weights, input)

        return self.state

