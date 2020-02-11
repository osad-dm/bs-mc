import numpy as np
import matplotlib.pyplot as plt
import datetime
import numpy
import scipy as sc
from qutip import *


'''Sx = sigmax()
Sy = sigmay()
Sz = sigmaz()
Sm = sigmam()
'''

A = Qobj([[1, 1], [1, 1]])*sc.sqrt(2)/2
A2 = tensor(Qobj([[1, 0], [0, 1]]), A)
A21 = Qobj([[1,0,0,0],[0,sc.sqrt(2)/2,sc.sqrt(2)/2,0],[0,sc.sqrt(2)/2,sc.sqrt(2)/2,0],[0,0,0,1]])
A21.dims = [[2,2], [2, 2]]
#A3 = tensor(Qobj([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), A)
#A21 = Qobj()
phi = basis(2)
phi2 = tensor(basis(2, 0), basis(2, 1))
#phi3 = tensor(basis(3), basis(2, 1))

'''times = np.linspace(0.0, 10.0, 20.0)
result = mesolve(A2, phi2, times, [], [])
print(result.states)
print("__________________________________")'''
print(A2)

print(A21)

print(phi2)
ans = A2*A21*A2
'''for i in ans:
    print(i**2)'''
print(ans)



#plt.show()

