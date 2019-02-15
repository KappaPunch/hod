import scipy as sp
from scipy.integrate import trapz, cumtrapz, romb
import numpy as np
import time

num = np.power(2., 25)+1
x = np.linspace(0., 10., num=num)
y = x * x

f = np.power(x, 3.) / 3.
f_inte = f[-1] - f[0]

st1 = time.time()
t = trapz(y,x)
print("trapz","%f"%(time.time()-st1))
print("diff. from true values", t-f_inte)

st2 = time.time()
dx = x[1]-x[0]
r = romb(y,dx=dx)
print("romb","%f"%(time.time()-st2))
print("diff. from true values", r-f_inte)

st3 = time.time()
c = cumtrapz(y,dx=dx)[-1]
print("cumtrapz","%f"%(time.time()-st2))
print("diff. from true values", c-f_inte)
