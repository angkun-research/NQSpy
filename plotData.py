import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# # J2s from 0.5 to 1.5 with step 0.1
# J2s = np.arange(0.5, 1.6, 0.1)
# Fids0 = [0.951,0.962,0.975,0.989,0.997,0.999982,0.991,0.979,0.935,0.911,0.861]
# Fids1 = [0.982,0.987,0.988,0.997,0.998,0.999989,0.998,0.981,0.966,0.976,0.970]
# Fids2 = [0.998,0.997,0.990,0.996,0.998,0.999979,0.999,0.988,0.985,0.989,0.982]

# Fids3 = [0.9875,0.9890,0.9928,0.9968,0.9989,0.999940,0.9982,0.9938,0.9866,0.9757,0.9652]
# Fids4 = [0.9939,0.9936,0.9954,0.9976,0.9989,0.999965,0.9990,0.9961,0.9917,0.9934,0.9831]
# Fids5 = [0.9983,0.9983,0.9969,0.9990,0.9994,0.999934,0.9983,0.9982,0.9968,0.9989,0.9963]

# if __name__ == "__main__":
# 	# plt.figure(figsize=(7,4.5))
# 	# plt.plot(J2s, 1 - np.array(Fids0), '-o', label=r'Exact NN')
# 	# plt.plot(J2s, 1 - np.array(Fids1), '-s', label=r'Perturbative $D_{hid}=32$')
# 	# plt.plot(J2s, 1 - np.array(Fids2), '-^', label=r'Perturbative $D_{hid}=64$')
# 	# plt.xlabel(r'$J_2$')
# 	# plt.ylabel(r'1 - Fidelity')
# 	# plt.yscale('log')
# 	# plt.grid(False)
# 	# plt.legend(loc='best')
# 	# #plt.title('Fidelities vs J2')
# 	# plt.tight_layout()
# 	# outname = '/Users/angkunwu/Desktop/fidelities.png'
# 	# plt.savefig(outname, dpi=300)
# 	# #print('Saved', outname)
# 	# plt.show()
	
#     plt.figure(figsize=(7,4.5))
#     plt.plot(J2s, 1 - np.array(Fids0), '-o', label=r'$k=2,h=32$')
#     plt.plot(J2s, 1 - np.array(Fids3), '-s', label=r'$k=3,h=32$')
#     plt.plot(J2s, 1 - np.array(Fids4), '-^', label=r'$k=5,h=32$')
#     plt.plot(J2s, 1 - np.array(Fids5), '-^', label=r'$k=5,h=64$')
#     plt.xlabel(r'$J_2$')
#     plt.ylabel(r'1 - Fidelity')
#     plt.yscale('log')
#     plt.grid(False)
#     plt.legend(loc='best')
# 	#plt.title('Fidelities vs J2')
#     plt.tight_layout()
#     outname = '/Users/angkunwu/Desktop/fidelities.png'
#     plt.savefig(outname, dpi=300)
# 	#print('Saved', outname)
#     plt.show()


# Ls = [11,21,31] #21
# DHs = [2772,3879876,4.81e+9]
# t2 = 0.5
# hidden_dim = 16
# E_exact = [-2.42988202, -2.519684,-2.541281]
# Samples = [64*128, 64*64, 64*128]
# def energy_smooth(Es, window=10):
# 	smoothed = []
# 	for i in range(0, len(Es), window):
# 		smoothed.append(np.mean(Es[i:i+window]))
# 	return smoothed

# #Es_smooth = energy_smooth(Es, window=20)
# Es_all = []
# Errs = []
# hidds = [16,16,16]
# for L, hidden_dim in zip(Ls, hidds):
# 	# csv_path = f"data/energy_record_L{L}_t2{t2}_hidden{hidden_dim}.csv"
# 	# Es = pd.read_csv(csv_path)['Energy'].values
# 	# Es_smooth = energy_smooth(Es, window=20)
# 	# Es_all.append(Es_smooth)
# 	csv_path = f"data/energy_error_exact_L{L}_t2{t2}_hidden{hidden_dim}_kernel2.csv"
# 	Es = pd.read_csv(csv_path)['Energy'].values
# 	err = pd.read_csv(csv_path)['Error'].values
# 	Es_all.append(Es)
# 	Errs.append(err)

# #xs = np.arange(len(Es_all[0])) * 20 
# plt.figure(figsize=(6,4))
# for k in range(len(Ls)):
# 	label = r"$L=%d,\ D_H=%.2e,\ E_{\mathrm{exact}}=%.3f$" % (Ls[k], DHs[k], E_exact[k])
# 	#line, = plt.plot(np.arange(len(Es_all[k])),Es_all[k], '-o', label=label)
# 	inds = np.arange(0, len(Es_all[k]), 2)
# 	line = plt.errorbar(
#         inds, #np.arange(len(Es_all[k][inds])),
#         Es_all[k][inds],
#         yerr=Errs[k][inds],          # per-point vertical error bars
#         fmt='-o',              # line + circle markers
#         capsize=2,             # little cap on bar ends
#         elinewidth=1,markersize=3,label=label
#         )
# 	plt.axhline(E_exact[k], color=line[0].get_color(), linestyle='--')
# plt.xlabel(r'VMC Step')
# plt.ylabel(r'Energy')
# plt.legend(loc='best')
# plt.xlim(0,400)#len(inds))
# #plt.xlim(0, len(Es_all[0])*20)
# plt.grid(False)
# plt.tight_layout()
# # outname = '/Users/angkunwu/Desktop/vmc_exact.png'
# # plt.savefig(outname, dpi=300)
# plt.show()


# L = 13, t2=0.5, t1=1.0, J1=1.0
# J2s = np.arange(0.0, 2.1, 0.1)
# Chis = [83,79,76,77,81,81,81,82,82,75,3,88,104,114,126,134,143,145,145,150,153] # cutoff 1e-12
# Fidelity_exact = [0.85429634,0.86922009,0.88480350,0.90104827,0.91789986,0.93519768,0.95258902,0.96938821,0.98436978,
#                   0.99553181, 1.00000000-1e-10,0.99443839,0.97626674,0.94517119,0.90350522,0.85503800,0.80332282,0.75094885,
#                   0.69963718,0.65057761,0.60464161]
# Fidelity_k2h16 = [0.84533238,0.85703021,0.88131058,0.88393199,0.90132892,0.91952372,0.93879968,0.97034514,0.97897542,
#                   0.98395431,0.99999613,0.99129796,0.96539015,0.92030489,0.86048895,0.80265188,0.72119343,0.64492208,
#                   0.60835588,0.52722341,0.47150242]
# Fidelity_k5h16 = [0.99329001,0.96445382,0.96489578,0.96582770,0.97432876,0.97762477,0.98422825,0.98931849,0.99403274,
#                   0.99794006,0.99997854,0.99692392,0.98586422,0.98796952,0.98777372,0.97808599,0.92674130,0.92714155,
#                   0.93647653,0.91792113,0.79480302]
# Fidelity_k5h64 = [0.99715596,0.99452996,0.99693811,0.99748856,0.99633789,0.99677116,0.99773496,0.99893361,0.99869370,
#                   0.99965948,0.99999875,0.99929082,0.99820292,0.99865788,0.99777848,0.99333668,0.99503058,0.99408579,
#                   0.98930866,0.99189758,0.98635614]

# if __name__ == "__main__":
#     fig, ax1 = plt.subplots(figsize=(6,4))

#     # left axis: 1 - Fidelity (log scale)
#     ax1.plot(J2s, 1 - np.array(Fidelity_exact), '-o', label=r'WF at $J_1=J_2=0$')
#     ax1.plot(J2s, 1 - np.array(Fidelity_k2h16), '-s', label=r'kernel$=2,D_{hid}=16$')
#     ax1.plot(J2s, 1 - np.array(Fidelity_k5h16), '-^', label=r'kernel$=5,D_{hid}=16$')
#     ax1.plot(J2s, 1 - np.array(Fidelity_k5h64), '-x', label=r'kernel$=5,D_{hid}=64$')
#     ax1.set_xlabel(r'$J_2$')
#     ax1.set_ylabel(r'1 - $|\langle \Psi_{\mathrm{ed}}|\Psi_{\mathrm{test}}\rangle|$')
#     ax1.set_yscale('log')
#     ax1.set_ylim(1e-5, 0.6)  # Adjust y-axis limits for better visualization
#     ax1.grid(False)
#     ax1.legend(loc='lower left')
#     # right axis: Chis (bond dimension)
#     ax2 = ax1.twinx()
#      # black color
#     ax2.plot(J2s, Chis, marker='o', color='black',linestyle='--',label=r'DMRG cutoff $10^{-12}$') # black color
#     ax2.set_ylabel(r'Bond dimension $\chi$')
#     ax2.legend(loc='lower right')
#     ax2.set_ylim(0, 160)  
#     plt.tight_layout()
#     # outname = '/Users/angkunwu/Desktop/nonexactJs.png'
#     # plt.savefig(outname, dpi=300)
#     plt.show()



# Final plot system size scaling, J2=0.9,t2=0.5
# Ls = [5,7,9,11,13,15,17,19]
# # cutoff 1e-12
# Chis = [8,19,38,54,75,91,99,107]
# MPSparams = [64,298,1204,3288,7556,13344,20915,29821]
# Gaps = [1.0362295913002155,0.7292629978716909,0.5074722973133694,0.3711837290202986,
#         0.2823491878517128,0.22155926848028606,0.1782730095235543,0.1464246152814095]
# GapsExact = [1.036483828388874,0.7273239046083431,0.5070419481698378,0.37144298233760065,
#              0.28293164164842644,0.22228443045130675,0.17904748620487787,0.14719909087218674]
# # cutoff min 1- fidelity 1e-3 (10 runs), lr = 1e-3, relu, epoch 10000 or avg loss < 1e-7
# FChiddim = [8,13,26,68,190,512]
# FCparams = [249,573,1691,7821,46551,294401]
# FChidL15 = [512,1024]
# FCparamL15 = [294401, 1113089]
# FidL15 = [0.98680770,0.99310958]
# FCparamsL15 = [249,573,1691,7821,46551,1113089]
# # Physical NN
# Kernals = [2,3,3,4,5,7,9]
# Dhidden = [4,4,6,8,8,12,16]
# Physicalparams = [81,105,181,305,353,697,1153]
# PhysicalFids = [0.99999667,0.99950284,0.99936825,0.99900764,0.99897051,0.99835217,0.99843025]

# if __name__ == "__main__":
#     fig, ax1 = plt.subplots(figsize=(6,4))

#     # left axis: 1 - Fidelity (log scale)
#     ax1.plot(Ls[:len(Physicalparams)], Physicalparams, '-o', label=r'VBS NN')
#     ax1.plot(Ls[:len(FCparams)], FCparams, '-s', label=r'FCNN')
#     ax1.plot(Ls[:len(FCparamsL15)], FCparamsL15, 's', color='C1', linestyle='--', label=None)
#     ax1.plot(Ls[:len(MPSparams)], MPSparams, '-^', label=r'MPS')
#     ax1.set_xlabel(r'$L$')
#     ax1.set_ylabel(r'Number of parameters')
#     ax1.set_yscale('log')
#     ax1.grid(False)
#     ax1.legend(loc='upper right')
#     # right axis: gap size scaling
#     ax2 = ax1.twinx()
#      # black color
#     ax2.plot(Ls, Gaps, marker='o', color='black',linestyle='--',label=r'$\Delta$') # black color
#     ax2.set_ylabel(r'many-body gap $\Delta$')
#     ax2.legend(loc='lower center')
#     ax2.set_yscale('log')
#     #ax2.set_ylim(0, 160)  
#     plt.tight_layout()
#     # outname = '/Users/angkunwu/Desktop/scalingLs.png'
#     # plt.savefig(outname, dpi=300)
#     plt.show()





Ls = [11,21,31] #[11,15,21] 
DHs = [2772,3879876,4.81e+9] #[2772,51480,3879876]
t2 = 0.5
E_exact = [-2.80219475,-3.26788669,-3.66565952]#[-2.80219475, -3.00731899, -3.26788669]
Samples = [64*128, 128*128, 128*128] # 64*128
J1 = 0.1
J2 = 0.09

Es_all = []
Errs = []
hidds = [32,32,32]
for L, hidden_dim in zip(Ls, hidds):
	csv_path = f"data/energy_error_nonexact_L{L}_t2{t2}_J1{J1}_J2{J2}_hidden{hidden_dim}_kernel5.csv"
	Es = pd.read_csv(csv_path)['Energy'].values
	err = pd.read_csv(csv_path)['Error'].values
	Es_all.append(Es)
	Errs.append(err)

#xs = np.arange(len(Es_all[0])) * 20 
plt.figure(figsize=(6,4))
for k in range(len(Ls)):
	label = r"$L=%d,\ D_H=%.2e,\ E_{\mathrm{exact}}=%.3f$" % (Ls[k], DHs[k], E_exact[k])
	#line, = plt.plot(np.arange(len(Es_all[k])),Es_all[k], '-o', label=label)
	inds = np.arange(0, len(Es_all[k]), 1)
	line = plt.errorbar(
        inds, #np.arange(len(Es_all[k][inds])),
        Es_all[k][inds],
        yerr=Errs[k][inds],          # per-point vertical error bars
        fmt='-o',              # line + circle markers
        capsize=2,             # little cap on bar ends
        elinewidth=1,markersize=3,label=label
        )
	plt.axhline(E_exact[k], color=line[0].get_color(), linestyle='--')
plt.xlabel(r'VMC Step')
plt.ylabel(r'Energy')
plt.legend(loc='best')
plt.title(r'$J_1=0.1,\ J_2=0.09$')
plt.xlim(0,200)
#plt.xlim(0, len(Es_all[0])*20)
plt.grid(False)
plt.tight_layout()
# outname = '/Users/angkunwu/Desktop/vmc_nonexact.png'
# plt.savefig(outname, dpi=300)
plt.show()