import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import curve_fit

x = 1.5,2,2.5,3,3.5,4,4.5,5,5.5,6
y =242,182,150,129,116,106,98.9,93.3,90,88

def func(x, a, b, c , d):
    return d + (a - d)/(1.+(x/c)**b)

#tck = interpolate.splrep(x, y, k=1)
xnew = np.arange(1.5, 25., .5)
#ynew = interpolate.splev(xnew, tck, der=0)

popt, pcov = curve_fit(func, x, y)

print(popt, pcov)
# for num in ynew:
#     print(num)

plt.figure()
plt.plot(x, y, 'x', xnew, ynew, 'b')
plt.legend(['Linear', 'Cubic Spline'])
#plt.axis([-0.05, 6.33, -1.05, 1.05])
plt.title('Cubic-spline interpolation')
plt.show()
