from qutip import *
import matplotlib.pyplot as plt
import numpy as np
import random
import time
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
    s_c1 = np.arcsin(np.sqrt(np.abs(0.5 + random.gauss(0, (C_err*0.5)/3))))
    s_c2 = np.arcsin(np.sqrt(np.abs(0.5 + random.gauss(0, (C_err*0.5)/3))))
    s_c3 = np.arcsin(np.sqrt(np.abs(0.5 + random.gauss(0, (C_err*0.5)/3))))

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
    result_3 = mesolve(qws1.H, result_2.states[1], times, col_ops)
    result_4 = mesolve(qws2.H, result_3.states[1], times, col_ops)
    result_5 = mesolve(qws1.H, result_4.states[1], times, col_ops)
    result_6 = mesolve(qws2.H, result_5.states[1], times, col_ops)
    # ress3 = mesolve(qws1.H, ress2.states[1], times, col_ops)
    result = result_6.states[1]

    return result_1.states[1], result_2.states[1], result_3.states[1], result_4.states[1], result_5.states[1], result_6.states[1]


initial_state = [0, 1, 0, 0]
N_MC = 1000


num_wg = len(initial_state)
dim = sum(initial_state) + 1
psi_vector = np.zeros(num_wg)
psi_vector = psi_vector.astype(int)
psi_vector = [i for i in initial_state]


gamma_list = [i * 0.1 for i in range(0, 9)]
se_list = [i * 0.05 for i in range(0, 11)]
start = time.time()
with open('sfid_'+str(initial_state)+'.txt', 'w') as f:
    f.writelines('seg\tgamma\tC_err\tav\tmax\tmin\n')
with open('sfid_'+str(initial_state)+'_raw.txt', 'w') as f:
    f.writelines('')

ideal_case = bs_me_solve(psi_vector, dim, num_wg, 0, 0)
for i in gamma_list:
    for j in se_list:
        fid = []
        for k in range(0, N_MC):
            result_bs = bs_me_solve(psi_vector, dim, num_wg, C_err=j, gamma=i)
            #final_result = [fock_trans(i.diag(), dim, num_wg) for i in result_bs]
            fid.append([fidelity(ideal_case[p], r) for p, r in enumerate(result_bs)])
        summer =[0, 0, 0, 0, 0, 0]
        maximum = [-1, -1, -1, -1, -1, -1]
        minimum = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
        for res in fid:
            for k, res_seg in enumerate(res):
                summer[k] += res_seg
                maximum[k] = max(maximum[k], res_seg)
                minimum[k] = min(minimum[k], res_seg)
        with open('sfid_'+str(initial_state)+'.txt', 'a') as f:
            for k in range(0, len(summer)):
                f.writelines(str(k+1)+'\t'+str(i) + '\t' + str(j) + '\t' + str(summer[k] / N_MC) + '\t' + str(maximum[k]) + '\t' + str(minimum[k])+'\n')
        with open('sfid_'+str(initial_state)+'_raw.txt', 'a') as f:
            f.writelines(str(fid)+'\n')
            #print(str(fid))

end = time.time()
with open('time_'+str(initial_state)+'_raw.txt', 'w') as f:
    f.writelines(str(end-start))

'''#print(sum(fid)/float(len(fid)))
fig, axes = plt.subplots()
axes.violinplot(fid)
#plt.plot(range(0, N_MC), fid)
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
plt.show()'''

