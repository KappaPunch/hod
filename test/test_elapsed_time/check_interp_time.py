import numpy as np
from scipy.interpolate import UnivariateSpline, CubicSpline, BSpline
import time

x = np.arange(1e3)
y = np.sin(x)
xs = np.linspace(-0.5, 9.6, num=1e5)

st = time.time()
cs = CubicSpline(x, y)
ys = cs(xs)
print("cubicspline","%f"%(time.time()-st))

st = time.time()
cs = UnivariateSpline(x, y)
ys = cs(xs)
print("univariate spline","%f"%(time.time()-st))

st = time.time()
cs = BSpline(x, y, k=3)
ys = cs(xs)
print("b-spline","%f"%(time.time()-st))
