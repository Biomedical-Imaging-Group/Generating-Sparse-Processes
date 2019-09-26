import numpy as np
import matplotlib.pyplot as plt
from .laws import gaussian, alpha_stable, laplace, gamma

class white_noise():

    def __init__(self, law_name, params):
        self.available_laws = {'gaussian': gaussian,
                          'alpha_stable': alpha_stable,
                          'laplace': laplace,
                          'gamma':gamma}
        self.law_name = law_name
        self.params = params


    def impulse_approx(self, lmda, T):
        N = np.random.poisson(lmda*T)

        knots = list(np.random.uniform(0, T, size = N))

        law = self.available_laws[self.law_name](self.params)
        law.lambda_effect(lmda)
        A_lst = law.jumps(N)

        return list(zip(knots, A_lst))

    def display(self, lmda, T):
        w = self.impulse_approx(lmda, T)

        knots = [element[0] for element in w]
        jump = [element[1] for element in w]

        plt.vlines(knots, 0, jump)
        plt.ylim(-5, 5)
        plt.axhline(0, color='k')
        plt.show()
