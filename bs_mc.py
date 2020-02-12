from qutip import *
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy as sc
#git хабу привет


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
    state_list = []
    for num, val in enumerate(q):
        if val != 0:
            state_list.append((num_convert(num, dim, n), val))
    return state_list


def num_convert(num, base, l):
    str_num = ''
    while num > 0 or len(str_num)<l:
        str_num = str(num % base) + str_num
        num //= base
    return(str_num)


def bs_me_solve(entry_state, f_dim, n_wg, C_err, gamma):
    err_bound = 0.5 * C_err
    s_c1 = np.arcsin(np.sqrt(random.uniform(0.5 - err_bound, 0.5 + err_bound)))
    s_c2 = np.arcsin(np.sqrt(random.uniform(0.5 - err_bound, 0.5 + err_bound)))
    s_c3 = np.arcsin(np.sqrt(random.uniform(0.5 - err_bound, 0.5 + err_bound)))

    hm1 = np.array([[   0, s_c1,    0,    0],
                    [s_c1,    0,    0,    0],
                    [   0,    0,    0, s_c2],
                    [   0,    0, s_c2,    0]])

    hm2 = np.array([[1,    0,    0,  0],
                    [0,    0, s_c3,  0],
                    [0, s_c3,    0,  0],
                    [0,    0,    0,  1]])

    psi0 = psi_init(entry_state, f_dim)
    qws1 = System(hm1, psi0, f_dim)
    qws2 = System(hm2, psi0, f_dim)
    times = np.linspace(0, 1, 2)

    col_ops = qws1.gen_c_ops(gamma)
    result_1 = mesolve(qws1.H, psi0, times, col_ops)
    result_2 = mesolve(qws2.H, result_1.states[1], times, col_ops)
    # ress3 = mesolve(qws1.H, ress2.states[1], times, col_ops)
    result = result_2.states[1]

    return result



initial_state = [1, 3, 0, 1]
N_MC = 10


num_wg = len(initial_state)
dim = sum(initial_state) + 1
psi_vector = np.zeros(num_wg)
psi_vector = psi_vector.astype(int)
psi_vector = [i for i in initial_state]
#fid = []
fid_1 = []

ideal_case = bs_me_solve(psi_vector, dim, num_wg, 0, 0)
for j in range(0, 1):
    fid = []
    for i in range(0, N_MC):
        result_bs = bs_me_solve(psi_vector, dim, num_wg, 0.1, 0)
        final_result = fock_trans(result_bs.diag(), dim, num_wg)
        fid.append(fidelity(ideal_case, result_bs))
    print('Fidelity:', sum(fid)/float(len(fid)))
    fid_1.append(sum(fid)/float(len(fid)))

#print(sum(fid)/float(len(fid)))
plt.plot(range(0, N_MC), fid)
plt.show()



final_result = fock_trans(ideal_case.diag(), dim,num_wg)
states_amps = []
ticks = []
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

