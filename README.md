# Generating Sparse Stochastic Processes

This library implements a method to generate sparse stochastic processes as defined [here](http://www.sparseprocesses.org/).

## Requirements

Before running the code please make sure that all the necessary requirements are installed with the command

```bash
pip install -r requirements.txt
```


### Tutorial

A detailed example is provided in the Jupyter Notebook `Example.ipynb`.


Here is a short example on how to use the library to simulate Brownian motion :


```python
from lib.lspline import L_spline
from lib.loperator import Operator
from lib.white_noise import white_noise

w = white_noise('gaussian', params=(0,1))
L = Operator([1, 0])
s = L_spline(L, w)

s.set_lambda(lmda = 100)

s.sample(T=1)
grid_values = s.get_grid_samples(T=1, step=0.01)
```

![png](brow.png)

