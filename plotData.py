import numpy as np
import matplotlib.pyplot as plt

# J2s from 0.5 to 1.5 with step 0.1
J2s = np.arange(0.5, 1.6, 0.1)
Fids0 = [0.951,0.962,0.975,0.989,0.997,0.999982,0.991,0.979,0.935,0.911,0.861]
Fids1 = [0.982,0.987,0.988,0.997,0.998,0.999989,0.998,0.981,0.966,0.976,0.970]
Fids2 = [0.998,0.997,0.990,0.996,0.998,0.999979,0.999,0.988,0.985,0.989,0.982]

Fids3 = [0.9875,0.9890,0.9928,0.9968,0.9989,0.999940,0.9982,0.9938,0.9866,0.9757,0.9652]
Fids4 = [0.9939,0.9936,0.9954,0.9976,0.9989,0.999965,0.9990,0.9961,0.9917,0.9934,0.9831]
Fids5 = [0.9983,0.9983,0.9969,0.9990,0.9994,0.999934,0.9983,0.9982,0.9968,0.9989,0.9963]

if __name__ == "__main__":
	# plt.figure(figsize=(7,4.5))
	# plt.plot(J2s, 1 - np.array(Fids0), '-o', label=r'Exact NN')
	# plt.plot(J2s, 1 - np.array(Fids1), '-s', label=r'Perturbative $D_{hid}=32$')
	# plt.plot(J2s, 1 - np.array(Fids2), '-^', label=r'Perturbative $D_{hid}=64$')
	# plt.xlabel(r'$J_2$')
	# plt.ylabel(r'1 - Fidelity')
	# plt.yscale('log')
	# plt.grid(False)
	# plt.legend(loc='best')
	# #plt.title('Fidelities vs J2')
	# plt.tight_layout()
	# outname = '/Users/angkunwu/Desktop/fidelities.png'
	# plt.savefig(outname, dpi=300)
	# #print('Saved', outname)
	# plt.show()
	
    plt.figure(figsize=(7,4.5))
    plt.plot(J2s, 1 - np.array(Fids0), '-o', label=r'$k=2,h=32$')
    plt.plot(J2s, 1 - np.array(Fids3), '-s', label=r'$k=3,h=32$')
    plt.plot(J2s, 1 - np.array(Fids4), '-^', label=r'$k=5,h=32$')
    plt.plot(J2s, 1 - np.array(Fids5), '-^', label=r'$k=5,h=64$')
    plt.xlabel(r'$J_2$')
    plt.ylabel(r'1 - Fidelity')
    plt.yscale('log')
    plt.grid(False)
    plt.legend(loc='best')
	#plt.title('Fidelities vs J2')
    plt.tight_layout()
    outname = '/Users/angkunwu/Desktop/fidelities.png'
    plt.savefig(outname, dpi=300)
	#print('Saved', outname)
    plt.show()
