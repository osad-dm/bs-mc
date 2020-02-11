import numpy as np
from qutip import *
import math
import cmath
import random
import matplotlib.pyplot as plt
import time


def err_snot(C):
    return Qobj([[cmath.exp(1j*math.pi/2.0), 0.0],[0.0 ,1.0]])*Qobj([[math.sqrt(1-C), 1j*math.sqrt(C)],[1j*math.sqrt(C) ,math.sqrt(1-C)]])*Qobj([[cmath.exp(1j*3*math.pi/2), 0],[0 ,cmath.exp(1j*math.pi)]])


state_00 = []
state_00.append(tensor(basis(2,0),basis(2,0)))
state_00.append(tensor(snot(),snot())*state_00[-1])
state_00.append(csign()*state_00[-1])
state_00.append(tensor(snot(),snot())*state_00[-1])
state_00.append(tensor(sigmaz(),sigmaz())*state_00[-1])
state_00.append(csign()*state_00[-1])
state_00.append(tensor(sigmaz(),sigmaz())*state_00[-1])
state_00.append(tensor(snot(),snot())*state_00[-1])
print(state_00[-1])

state_10 = []
state_10.append(tensor(basis(2,0),basis(2,0)))
state_10.append(tensor(snot(),snot())*state_10[-1])
state_10.append(csign()*state_10[-1])
state_10.append(tensor(snot(),snot())*state_10[-1])
state_10.append(tensor(sigmaz(),qeye(2))*state_10[-1])
state_10.append(csign()*state_10[-1])
state_10.append(tensor(sigmaz(),sigmaz())*state_10[-1])
state_10.append(tensor(snot(),snot())*state_10[-1])
print(state_10[-1])

state_01 = []
state_01.append(tensor(basis(2,0),basis(2,0)))
state_01.append(tensor(snot(),snot())*state_01[-1])
state_01.append(csign()*state_01[-1])
state_01.append(tensor(snot(),snot())*state_01[-1])
state_01.append(tensor(qeye(2),sigmaz())*state_01[-1])
state_01.append(csign()*state_01[-1])
state_01.append(tensor(sigmaz(),sigmaz())*state_01[-1])
state_01.append(tensor(snot(),snot())*state_01[-1])
print(state_01[-1])

state_11 = []
state_11.append(tensor(basis(2,0),basis(2,0)))
state_11.append(tensor(snot(),snot())*state_11[-1])
state_11.append(csign()*state_11[-1])
state_11.append(tensor(snot(),snot())*state_11[-1])
state_11.append(tensor(qeye(2),qeye(2))*state_11[-1])
state_11.append(csign()*state_11[-1])
state_11.append(tensor(sigmaz(),sigmaz())*state_11[-1])
state_11.append(tensor(snot(),snot())*state_11[-1])
print(state_11[-1])

N_MC = 100  # Number of Monte-Carlo runs
C_err = 70  # Error of splitting coefficient in %
fid_00 = []
fid_10 = []
fid_01 = []
fid_11 = []

for i in range(0, N_MC):
    state_err_00 = []
    state_err_00.append(tensor(basis(2, 0), basis(2, 0)))
    state_err_00.append(tensor(err_snot(random.uniform(0.5 - 0.5 * C_err / 100, 0.5 + 0.5 * C_err / 100)),
                               err_snot(random.uniform(0.5 - 0.5 * C_err / 100, 0.5 + 0.5 * C_err / 100))) *
                        state_err_00[-1])
    state_err_00.append(csign() * state_err_00[-1])
    state_err_00.append(tensor(err_snot(random.uniform(0.5 - 0.5 * C_err / 100, 0.5 + 0.5 * C_err / 100)),
                               err_snot(random.uniform(0.5 - 0.5 * C_err / 100, 0.5 + 0.5 * C_err / 100))) *
                        state_err_00[-1])
    state_err_00.append(tensor(sigmaz(), sigmaz()) * state_err_00[-1])
    state_err_00.append(csign() * state_err_00[-1])
    state_err_00.append(tensor(sigmaz(), sigmaz()) * state_err_00[-1])
    state_err_00.append(tensor(err_snot(random.uniform(0.5 - 0.5 * C_err / 100, 0.5 + 0.5 * C_err / 100)),
                               err_snot(random.uniform(0.5 - 0.5 * C_err / 100, 0.5 + 0.5 * C_err / 100))) *
                        state_err_00[-1])
    fid_00.append(fidelity(state_00[-1], state_err_00[-1]))

    state_err_10 = []
    state_err_10.append(tensor(basis(2, 0), basis(2, 0)))
    state_err_10.append(tensor(err_snot(random.uniform(0.5 - 0.5 * C_err / 100, 0.5 + 0.5 * C_err / 100)),
                               err_snot(random.uniform(0.5 - 0.5 * C_err / 100, 0.5 + 0.5 * C_err / 100))) *
                        state_err_10[-1])
    state_err_10.append(csign() * state_err_10[-1])
    state_err_10.append(tensor(err_snot(random.uniform(0.5 - 0.5 * C_err / 100, 0.5 + 0.5 * C_err / 100)),
                               err_snot(random.uniform(0.5 - 0.5 * C_err / 100, 0.5 + 0.5 * C_err / 100))) *
                        state_err_10[-1])
    state_err_10.append(tensor(sigmaz(), qeye(2)) * state_err_10[-1])
    state_err_10.append(csign() * state_err_10[-1])
    state_err_10.append(tensor(sigmaz(), sigmaz()) * state_err_10[-1])
    state_err_10.append(tensor(err_snot(random.uniform(0.5 - 0.5 * C_err / 100, 0.5 + 0.5 * C_err / 100)),
                               err_snot(random.uniform(0.5 - 0.5 * C_err / 100, 0.5 + 0.5 * C_err / 100))) *
                        state_err_10[-1])
    fid_10.append(fidelity(state_10[-1], state_err_10[-1]))

    state_err_01 = []
    state_err_01.append(tensor(basis(2, 0), basis(2, 0)))
    state_err_01.append(tensor(err_snot(random.uniform(0.5 - 0.5 * C_err / 100, 0.5 + 0.5 * C_err / 100)),
                               err_snot(random.uniform(0.5 - 0.5 * C_err / 100, 0.5 + 0.5 * C_err / 100))) *
                        state_err_01[-1])
    state_err_01.append(csign() * state_err_01[-1])
    state_err_01.append(tensor(err_snot(random.uniform(0.5 - 0.5 * C_err / 100, 0.5 + 0.5 * C_err / 100)),
                               err_snot(random.uniform(0.5 - 0.5 * C_err / 100, 0.5 + 0.5 * C_err / 100))) *
                        state_err_01[-1])
    state_err_01.append(tensor(qeye(2), sigmaz()) * state_err_01[-1])
    state_err_01.append(csign() * state_err_01[-1])
    state_err_01.append(tensor(sigmaz(), sigmaz()) * state_err_01[-1])
    state_err_01.append(tensor(err_snot(random.uniform(0.5 - 0.5 * C_err / 100, 0.5 + 0.5 * C_err / 100)),
                               err_snot(random.uniform(0.5 - 0.5 * C_err / 100, 0.5 + 0.5 * C_err / 100))) *
                        state_err_01[-1])
    fid_01.append(fidelity(state_01[-1], state_err_01[-1]))

    state_err_11 = []
    state_err_11.append(tensor(basis(2, 0), basis(2, 0)))
    state_err_11.append(tensor(err_snot(random.uniform(0.5 - 0.5 * C_err / 100, 0.5 + 0.5 * C_err / 100)),
                               err_snot(random.uniform(0.5 - 0.5 * C_err / 100, 0.5 + 0.5 * C_err / 100))) *
                        state_err_11[-1])
    state_err_11.append(csign() * state_err_11[-1])
    state_err_11.append(tensor(err_snot(random.uniform(0.5 - 0.5 * C_err / 100, 0.5 + 0.5 * C_err / 100)),
                               err_snot(random.uniform(0.5 - 0.5 * C_err / 100, 0.5 + 0.5 * C_err / 100))) *
                        state_err_11[-1])
    state_err_11.append(tensor(qeye(2), qeye(2)) * state_err_11[-1])
    state_err_11.append(csign() * state_err_11[-1])
    state_err_11.append(tensor(sigmaz(), sigmaz()) * state_err_11[-1])
    state_err_11.append(tensor(err_snot(random.uniform(0.5 - 0.5 * C_err / 100, 0.5 + 0.5 * C_err / 100)),
                               err_snot(random.uniform(0.5 - 0.5 * C_err / 100, 0.5 + 0.5 * C_err / 100))) *
                        state_err_11[-1])
    fid_11.append(fidelity(state_11[-1], state_err_11[-1]))

print(fid_00)
plt.plot(range(0,N_MC),fid_00)
plt.show()
print(sum(fid_00)/float(len(fid_00)))
print(min(fid_00))
print(max(fid_00))