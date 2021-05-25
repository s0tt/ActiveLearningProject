import numpy as np 
import matplotlib.pyplot as plt 
import sys 


x, run_1_bald, run_2_bald, run_3_bald, run_4_bald, run_5_bald = np.loadtxt('f1_scores_bald.txt', usecols=(0, 1, 2, 3, 4, 5), unpack=True)
x, run_1, run_2, run_3, run_4, run_5 = np.loadtxt('f1_scores_random.txt', usecols=(0, 1, 2, 3, 4, 5), unpack=True)
x, run_1_entro, run_2_entro, run_3_entro, run_4_entro, run_5_entro = np.loadtxt('f1_scores_max_entropy.txt', usecols=(0, 1, 2, 3, 4, 5), unpack=True)
x, run_1_vari, run_2_vari, run_3_vari, run_4_vari, run_5_vari = np.loadtxt('f1_scores_max_variation.txt', usecols=(0, 1, 2, 3, 4, 5), unpack=True)
x, run_1_mean_std, run_2_mean_std, run_3_mean_std, run_4_mean_std, run_5_mean_std = np.loadtxt('f1_scores_mean_std.txt', usecols=(0, 1, 2, 3, 4, 5), unpack=True)
x = x - 2000


def get_average( run_1, run_2, run_3, run_4, run_5): 
    stacked_arrays = np.array([run_1, run_2, run_3, run_4, run_5])
    averaged_array = np.average(stacked_arrays,axis=0)
    return averaged_array

def plot_single_accuracy_with_five_runs(x, run_1, run_2, run_3, run_4, run_5):
    plt.figure()
    averaged_array = get_average(run_1, run_2, run_3, run_4, run_5)

    plt.plot(x, run_1, label='first_run')
    plt.plot(x, run_2, label='second_run')
    plt.plot(x, run_3, label='thrid_run')
    plt.plot(x, run_4, label='fourth_run')
    plt.plot(x, run_5, label='fifth_run')
    plt.plot(x, averaged_array, label="Average", linewidth=3)

    plt.legend()

mean_std_average = get_average(run_1_mean_std, run_2_mean_std, run_3_mean_std, run_4_mean_std, run_5_mean_std)
variation_average = get_average(run_1_vari, run_2_vari, run_3_vari, run_4_vari, run_5_vari)

bald_average = get_average(run_1_bald, run_2_bald, run_3_bald, run_4_bald, run_5_bald)
max_entropy_average = get_average(run_1_entro, run_2_entro, run_3_entro, run_4_entro, run_5_entro)
random_average = get_average(run_1, run_2, run_3, run_4, run_5)






font = {'family' : 'Times New Roman',
        'size'   : 10}
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text', usetex=True)
plt.rc('font', **font)
#plt.rc('font', family='Times New Roman')
plt.rc('legend',fontsize=10)
plt.rc('legend', labelspacing=.1)

fig = plt.figure(figsize=(3.3, 3.0))
plt.subplots_adjust(bottom=0.17, left = 0.10, right=0.9)

plt.plot(x, bald_average/10507, label="BALD")
plt.plot(x, max_entropy_average/10507, label="Max Entropy")
plt.plot(x, random_average/10507, label="Random")
plt.plot(x, mean_std_average/10507, label="Mean STD")
plt.plot(x, variation_average/10507, label="Var Ratios")
plt.xlabel("Number of queried samples")
plt.xlim(0, 2500)
plt.grid()
#plt.ylabel("Accuracy")

plt.legend()


plt.savefig("bert_evaluation.pdf" , format='pdf')

plt.show()