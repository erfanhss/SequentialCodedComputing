import numpy as np
import matplotlib.pyplot as plt
import scipy
#### Theoretical Performance Analysis
def choose(n, k):
  return scipy.math.factorial(n)/(scipy.math.factorial(k)*scipy.math.factorial(n-k))

def prob(ps, n, k, w):
  if n==1:
    if k>=1 and k<=w:
      return ps[k-1]
    else:
      return 0 
  if k<n:
    return 0
  res = 0
  for i in range(1, min(k-n+1, w)+1):
    res += ps[i-1]*prob(ps, n-1, k-i, w)
  return res
def L_func(p, T, W, s):
  ps = [choose(W, i)*(1-p)**i * p**(W-i) for i in range(W+1)]
  ps[-1] = ps[-1] + ps[0]
  ps = ps[1:]
  res = 0
  for i in range(T, s):
    res += prob(ps, T, i, W)*max(0, s-i)
  return res
def queue_average(s, p, w):
  p_vector = [choose(w, i)*(p**(w-i))*((1-p)**i) for i in range(w+1)]
  p_vector[-1] = p_vector[-1] + p_vector[0]
  p_vector[0] = 0
  f_values = np.zeros(s+1)
  for crt_s in range(1, s+1):
    if crt_s <= w:
      f_values[crt_s] = 1 + np.dot(p_vector[1:crt_s], (f_values[1:crt_s])[::-1])
    else:
      f_values[crt_s] = 1 + np.dot(p_vector[1:w+1], (f_values[crt_s-w:crt_s])[::-1])
  return f_values[-1]


def DIP(s, p, w, T, alpha, lam):
  p_c = np.sum([choose(w, i)*(p**(w-i))*((1-p)**i) for i in range(lam)])
  probs = [choose(T*w, i)*(p**(T*w-i))*((1-p)**i) for i in range(min(s-1, w*T))]
  L = L_func(p, T, w, s)
  thr = ((np.minimum(queue_average(s, p, w), T+1) + L/lam)*(alpha*p_c + (1-p_c)))/s 
  return thr

def uncoded(w, alpha, p):
  p_f = 1-(1-p)**w
  return p_f*alpha/w + (1-p_f)/w

def poly(w, alpha, epsilon, p):
  p_f = np.sum([choose(w, i)*(p**i)*((1-p)**(w-i)) for i in range(epsilon+1, w+1)])
  return p_f*alpha/(w-epsilon) + (1-p_f)/(w-epsilon)

def IDIP_lower_bound(j, w, alpha, p, N, T):
  res = 0
  for i in range(N+1, (T+1)*w+1):
    tmp = (p**i)*((1-p)**((T+1)*w-i))*choose((T+1)*w, i)
    res += tmp
  p_f = res
  return (alpha*p_f + (1-p_f))/(w - N/(T+1))

def DIP_infinite_delay(s, p, w, alpha):
  p_a = p**w
  # return (1-p_a+p_a*alpha)/(w)/(1-p+p**w)
  return queue_average(s, p, w)*((1-p_a) + p_a*alpha)/s


def find_best(j, w, alpha, p, type, max_tolerable_delay=10):
  if type == 'poly':
    crt_best = 1000000.
    for epsilon in range(1, w):
      crt = poly(w, alpha, epsilon, p)
      if crt < crt_best:
        crt_best = crt
    return crt_best
  if type == 'arbitrary':
    crt_best = 1000000.0
    for T in range(1, max_tolerable_delay):
      for N in range(1, (T+1)*w):
          crt = arbitrary_lower_bound(j, w, alpha, p, N, T)
          if crt < crt_best:
            crt_best = crt
    return crt_best
  if type == 'MI_inf':
    crt_best = 1000000.0
    for s in range(w, (max_tolerable_delay)*w):
      crt = MI_infinite_delay(s, p, w, alpha)
      if crt < crt_best:
        crt_best = crt
    return crt_best
  if type == 'model_independent':
    crt_best = 1000000.0
    for T in range(1, max_tolerable_delay):
      for lam in range(1, w+1):
        for s in range((T+1)*w):
          crt = model_independent(s, p, w, T, alpha, lam)
          if crt < crt_best:
            crt_best = crt
    return crt_best
