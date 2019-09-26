import numpy as np
from scipy.stats import levy_stable

class gaussian():

    def __init__(self, params):
        self.mu, self.sigma = params

    def lambda_effect(self, lmda):
        self.mu = self.mu/lmda
        self.sigma = self.sigma/lmda

    def jumps(self, N):

        return self.mu + np.sqrt(self.sigma)*np.random.randn(N)


class alpha_stable():

    def __init__(self, params):
        self.params = params

    def lambda_effect(self, lmda):
        alpha, beta, mu, c = self.params
        if alpha == 1:
            self.params = (alpha, beta, mu/lmda + 2/np.pi*c*beta*np.log(lmda)/lmda, c/(lmda))
        else:
            self.params = (alpha, beta, mu/lmda, c/(lmda**(1/alpha)))

    def jumps(self, N):
        return levy_stable.rvs(self.params[0], beta=0, scale=self.params[-1], size=N)

    def stable(self, alpha, beta, mu=0, c=1):
        return levy_stable.rvs(alpha, c)

class laplace():

    def __init__(self, params):
        self.mu, self.b = params
        self.lmda = 1

    def sample(self):
        gamma_1 = np.random.gamma(1.0/self.lmda, 1)
        gamma_2 = np.random.gamma(1.0/self.lmda, 1)
        return self.mu/self.lmda + self.b*(gamma_1 - gamma_2)

    def lambda_effect(self, lmda):
        self.lmda = lmda

    def jumps(self, N):
        return [self.sample() for i in range(N)]

class gamma():

    def __init__(self, params):
        self.alpha, self.beta = params

    def lambda_effect(self, lmda):
        self.alpha /= lmda

    def jumps(self, N):
        return [np.random.gamma(self.alpha, 1.0/self.beta) for i in range(N)]


class compound_poisson():

    def __init__(self, params):
        self.law, self.lmda = params

    def lambda_effect(self, lmda):
        self.lmda *= 1/lmda

    def sample(self):
        N = np.random.poisson(self.lmda)
        iid_jumps = [0] + self.law.jumps(N)
        return np.sum(iid_jumps)

    def jumps(self, N):
        return [self.sample() for i in range(N)]
