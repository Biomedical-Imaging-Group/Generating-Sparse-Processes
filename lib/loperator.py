import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import residue

import sympy
from sympy.functions import exp

class Operator():

    def __init__(self, P, Q=[1]):

        self.P = P
        self.Q = Q

        self.roots = np.roots(self.P)
        self.degree = len(self.P) - 1

        self.green_func = self.green_components()
        self.discretization_step = None
        self.b_support = None

    def heavside(self, x, alpha):
        def heav(x):
            if x >= 0:
                return 1
            return 0
        if np.real(alpha) <= 0:
            return heav(x)
        else:
            return -1*heav(-x)

    def base_func(self, coef, m, p):
        return lambda x : coef * x**(m-1)/np.math.factorial(m-1) * np.exp(p*x) * self.heavside(x, p)

    def green_components(self):
        mult = {}
        simple_decomp_coeffs, poles, k = residue(self.Q, self.P)
        components = []
        for i, p in enumerate(poles):
            if p in mult:
                mult[p] += 1
            else:
                mult[p] = 1
            m = mult[p]
            components.append((simple_decomp_coeffs[i], m, p))
        return components

    def green(self, x):
        val = 0
        for params in self.green_func:
            val += self.base_func(*params)(x)
        return np.real(val)

    def b_spline(self, x):
        val = 0
        norm = 1
        for params in self.green_func:
            for i in range(len(self.discrete_operator)):
                val += 1.0/norm * self.discrete_operator[i] * self.base_func(*params)(x - i * self.discretization_step)
        return np.real(val)

    def set_discretization_step(self, h):
        self.discretization_step = h
        self.b_spline_support = self.degree * h
        self.discrete_operator = self.discretize_me()

    def discretize_me(self):
        h = self.discretization_step
        X = sympy.Symbol('X')
        Ld = 1
        for alpha in self.roots:
            Ld = Ld * (1 - exp(alpha*h)*X)
        return list(map(complex, sympy.Poly(sympy.expand(Ld), X).coeffs()[::-1]))

    def discrete_inverse(self, u):
        inverse = np.concatenate([np.zeros(self.degree), np.zeros_like(u)])
        for k in range(self.degree, len(inverse)):
            inverse[k] = u[k - self.degree]
            for i in range(1, self.degree + 1):
                inverse[k] += -self.discrete_operator[i] * inverse[k - i]
        return inverse[self.degree:]
