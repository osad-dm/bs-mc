from qutip import *
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

import time


class System:
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


def fock_trans(q, dim, n):
    num = 0
    state_list = []
    for i in q:
        if i != 0:
            state_list.append((num, i))
        num = num+1
    base = dim
    new_num_list = []
    j = 0
    for i in state_list:
        new_num_list.append(['', i[1]])
        basisState = i[0]
        while basisState > 0 or len(new_num_list[j][0]) < n:
            new_num_list[j][0] = str(basisState % base) + new_num_list[j][0]
            basisState //= base
        j = j + 1

    return new_num_list


entry_state = [1, 0, 1, 0]
n = len(entry_state)
dim = sum(entry_state)
psi_vector = np.zeros(n)
psi_vector = psi_vector.astype(int)
psi_vector = [i for i in entry_state]
print(entry_state, ' ', len(entry_state), ' ', dim, ' ', psi_vector)

sq_2 = sc.sqrt(2)/2
s_c1 = np.arcsin(np.sqrt(0.3))
s_c2 = np.arcsin(np.sqrt(0.4))
s_c3 = np.arcsin(np.sqrt(0.5))

hm1 = np.array([[0,  s_c1,    0,    0],
                [s_c1,  0,    0,    0],
                [0,     0,    0, s_c2],
                [0,     0, s_c2,    0]])

hm2 = np.array([[1, 0,    0, 0],
                [0, 0, s_c3, 0],
                [0, s_c3, 0, 0],
                [0, 0,    0, 1]])


psi0 = psi_init(psi_vector, dim)
qws1 = System(hm1, psi0, dim)
qws2 = System(hm2, psi0, dim)
times = np.linspace(0, 1, 2)


col_ops = qws1.gen_c_ops(0)
result_1 = mesolve(qws1.H, psi0, times, col_ops)
#result_2 = mesolve(qws2.H, result_1.states[1], times, col_ops)
#ress3 = mesolve(qws1.H, ress2.states[1], times, col_ops)


diag_res = result_1.states[1].diag()


states_amps = []
ticks = []
final_result = fock_trans(diag_res, dim, n)
print(final_result)


sum_of_coefficients = 0
non_zero_states_count = 0
for i in final_result:
    states_amps.append(abs(i[1]))#**2)
    ticks.append(i[0])
    sum_of_coefficients = sum_of_coefficients + i[1]


print("Sum of coeff:")
print(sum_of_coefficients)
fig, ax = plt.subplots()
plt.bar(np.arange(1, len(states_amps)+1), states_amps)
plt.xticks(np.arange(1, len(states_amps)+1), ticks)
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




