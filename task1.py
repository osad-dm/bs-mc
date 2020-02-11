from qutip import *
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

import time
#import psutil

class system:
    def __init__(self, ham_matrix, psi_init, fock_dim):
        self.a = []
        for i in range(0, len(ham_matrix)):
            op_list = []
            for j in range(0, len(ham_matrix)):
                if j == i:
                    op_list.append(destroy(fock_dim))
                else:
                    op_list.append(qeye(fock_dim))
            self.a.append(tensor(op_list))

        for i in range(0, len(ham_matrix)):
            for j in range(0, len(ham_matrix)):
                if i == 0 and j == 0:
                    self.H = ham_matrix[0][0] * self.a[0].dag() * self.a[0]
                else:
                    self.H = self.H + ham_matrix[i][j] * self.a[i].dag() * self.a[j]

    def gen_c_ops(self, gamma):
        c_ops_list = []
        for i in range(0, len(self.a)):
            c_ops_list.append(gamma * self.a[i])
        return c_ops_list

    def gen_cormatrix_2(self, psi):
        cormatrix = np.zeros((len(self.a), len(self.a)), dtype=complex)
        for i in range(0, len(self.a)):
            for j in range(0, len(self.a)):
                temp_op = self.a[i].dag() * self.a[j].dag() * self.a[i] * self.a[j]
                cormatrix[i][j] = temp_op.matrix_element(psi.dag(), psi)
        return cormatrix

    def gen_cormatrix_2_loss(self, psi):
        cormatrix = np.zeros((len(self.a), len(self.a)), dtype=complex)
        for i in range(0, len(self.a)):
            for j in range(0, len(self.a)):
                temp_op = self.a[i].dag() * self.a[j].dag() * self.a[i] * self.a[j]
                temp_op_j = psi * temp_op
                cormatrix[i][j] = temp_op_j.tr()
        return cormatrix


def psi_init(psi_vector, fock_dim):
    psi_list = []
    for i in range(0,len(psi_vector)):
        psi_list.append(fock(fock_dim,psi_vector[i]))
    return tensor(psi_list)


def uniform_array(num_wg, b, k):
    ham_matrix = np.zeros((num_wg,num_wg))
    for i in range(0,num_wg):
        for j in range(0,num_wg):
            if i == j:
                ham_matrix[i][j] = b
            elif i == j+1 or j == i+1:
                ham_matrix[i][j] = k
    return ham_matrix


def fock_trans(q, n, m):
    num = 0
    state_list = []
    for i in q:
        if i != 0:
            state_list.append((num, i))
        num = num+1
    base = n
    newNum_list = []
    j = 0
    for i in state_list:
        newNum_list.append(['', i[1]])
        basisState = i[0]
        while basisState > 0 or len(newNum_list[j][0]) < m:
            newNum_list[j][0] = str(basisState % base) + newNum_list[j][0]
            basisState //= base
        j = j + 1

    return newNum_list


n = 4
dim = 3
psi_vector = np.zeros(n)
psi_vector = psi_vector.astype(int)
psi_vector[1] = 1
psi_vector[2] = 1
hm2 = np.array([[1,0,0,0],[0,sc.sqrt(2)/2,sc.sqrt(2)/2,0],[0,sc.sqrt(2)/2,sc.sqrt(2)/2,0],[0,0,0,1]])
hm3 = np.array([[sc.sqrt(2)/2,sc.sqrt(2)/2,0,0],[sc.sqrt(2)/2,sc.sqrt(2)/2,0,0],[0,0,sc.sqrt(2)/2,sc.sqrt(2)/2],[0,0,sc.sqrt(2)/2,sc.sqrt(2)/2]])

psi0 = psi_init(psi_vector, dim)
hm = uniform_array(n, 1, np.sqrt(2)/2)
qws = system(hm3, psi0, dim)
qws2 = system(hm2, psi0, dim)
times = np.linspace(0, np.pi/2, 2)
print(hm2)
print(hm3)

start = time.time()
ress  = sesolve(qws.H, psi0, times, [])
result_un = sesolve(qws2.H, ress.states[1], times, [])
print("lol")
for i in qws.gen_cormatrix_2(result_un.states[1]):
    print(abs(i**2))
print("kek")
print(qws.gen_cormatrix_2(result_un.states[1]))

amps=[]
ticks=[]
HEH = fock_trans(result_un.states[1], dim, n)
print(HEH)

sum = 0
for i in HEH:
    amps.append(abs(i[1][0][0])**2)
    ticks.append(i[0])


fig, ax = plt.subplots()
plt.bar(np.arange(1, len(amps)+1), amps)
plt.xticks(np.arange(1, len(amps)+1), ticks)
plt.ylim(0, 1)
plt.xlabel('States')
plt.ylabel('Probability')
plt.tight_layout()
plt.show()

''' fig, axes = plt.subplots(1, len(result_un.states))
for i in range(0, len(result_un.states)):
    print(i," vector")
    print(result_un.states[i])
    print("Fock", i)
    print(fock_trans(result_un.states[i], 2, 4))
    plot_fock_distribution(result_un.states[i], fig=fig, ax=axes[i], title="Fock state");

plt.show()'''


