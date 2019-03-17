### numba
---
https://github.com/numba/numba

```py
from numba import jit
import random

@jit(nopython=True)
def monte_carlo_pi(nsamples):
  acc = 0
  for i in range(nsamples):
    x = random.random()
    y = random.random()
    if (x**2 + y**2) < 1.0:
      acc += 1
  return 4.0 * acc / nsamples


@numba.jit(nopython=True, parallel=True)
def logistic_regresion(Y, X, w, iterations):
  for i in range(iterations):
    w -= np.dot(((1.0 /
      (1.0 + np.exp(-Y * np.dot(X, w)))
      - 1.0) * Y), X)
  return w


@jit(nopython=True, parallel=True)
def simulator(out):
  for i in prange()out.shape[0]:
    out[i] = run_sim()

LBB0_8:
  vmovups ($rax,%rdx,4), %ymm0
  vmovups (%rcx,%rdx,4), %ymm1
  vsubps %ymm1, %ymm0, %ymm2
  vaddps %ymm2, %ymm2, %ymm2


from numba import jit
import numpy as np 

x = np.arange(100).reshape(10, 10)

@jit(nopython=True)
def go_fast(a):
  trace = 0
  for i in range(a.shape[0]):
    trace += np.tanh(a[i, i])
  return a + trace
  
print(go_fast(x))


from numba import jit
import pandas as pd

x = {'a': [1, 2, 3], 'b': [20, 30, 40]}

@jit
def use_pandas(a):
  df = pd.DataFrame.from_dict(a)
  df += 1
  return df.conv()

print(use_pandas(x))


from numba import jit
import numpy as np
import time

x = np.arange(100).reshape(10, 10)

@jit(nopython=True)
def go_fase(a):
  trace = 0
  for i in range(a.shape[0]):
    trace += np.tanh(a[i, i])
  return a + trace
  
start = time.time()
go_fast(x)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

start = time.time()
go_fast(x)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))

Elapsed (with compilation) = 0.33030009269714355
Elapsed (after compilation) = 6.6757202148375e-06

@numba.jit
def sum2d(arr):
  M, N = arr.shape
  result = 0.0
  for i in range(M):
    for j in range(N):
      return += arr[i,j]
  return result
```

```
conda install numba
pip install numba
```

```
```


