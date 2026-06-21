import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import pandas as pd

plt.rcParams['font.family'] = 'serif'       # or 'sans-serif', 'monospace'
plt.rcParams['font.serif'] = ['Times New Roman']
# plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# Only spin results (no hopping)
# Ls = [12,14,16,18,20]
# # hid=16, kernel=2, train on 2,4,6,8,10
# Fids = [0.9999885559082031,0.9998733997344971,0.9992972612380981,0.9971129894256592,0.9914627075195312]
# FidsTrain = [0.99999964,0.99999958,0.99999952,0.99999726]
# # J2=0.0, hid=64, kernel=5
# FidsJ20 = [0.9179648756980896,0.7908716201782227,0.7399603128433228,0.6359143257141113,0.596321702003479]
# FidsJ20Train = [0.99956441,0.99786794,0.99332392,0.98385531]
# # J2=1.5 hid=64, kernel=5
# FidsJ215 = [0.6066146492958069,0.503864586353302,0.33894866704940796,0.2450920045375824,0.1748594343662262]
# FidsJ215Train = [0.99183154,0.98012489,0.96675253,0.94783098]

# if __name__ == "__main__":
#     fig, ax = plt.subplots(figsize=(6*0.85,4*0.85)) #(6,4)
#     line = ax.plot(Ls, np.array(Fids), '-o', label=r'$J_2=1.0$',markersize=4)
#     ax.plot(Ls[:len(FidsTrain)], np.array(FidsTrain), '--o', label=None, color=line[0].get_color(),markerfacecolor='none')
#     line = ax.plot(Ls, np.array(FidsJ20), '-s', label=r'$J_2=0.0$',markersize=4)
#     ax.plot(Ls[:len(FidsJ20Train)], np.array(FidsJ20Train), '--s', label=None, color=line[0].get_color(),markerfacecolor='none')
#     line = ax.plot(Ls, np.array(FidsJ215), '-^', label=r'$J_2=1.5$',markersize=4)
#     ax.plot(Ls[:len(FidsJ215Train)], np.array(FidsJ215Train), '--^', label=None, color=line[0].get_color(),markerfacecolor='none')
#     ax.set_xlabel(r'$L$')
#     ax.set_ylabel(r'$|\langle \Psi_{\mathrm{ED}}|\Psi_{\mathrm{NQS}}\rangle|$')
#     #ax.set_yscale('log')
#     ax.grid(False)
#     ax.legend(loc='lower left')
#     plt.tight_layout()
#     outname = '/Users/angkunwu/Desktop/extrapolateLs.png'
#     plt.savefig(outname, dpi=300)
#     plt.show()



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

# VMC exact case
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
# samples = [128*128, 128*128,128*128*2]
# for L, hidden_dim, sample in zip(Ls, hidds, samples):
# 	# csv_path = f"data/energy_record_L{L}_t2{t2}_hidden{hidden_dim}.csv"
# 	# Es = pd.read_csv(csv_path)['Energy'].values
# 	# Es_smooth = energy_smooth(Es, window=20)
# 	# Es_all.append(Es_smooth)
# 	csv_path = f"data/energy_error_exact_L{L}_t2{t2}_hidden{hidden_dim}_kernel2_sam{sample}.csv"
# 	#csv_path = f"data/energy_error_exact_L{L}_t2{t2}_hidden{hidden_dim}_kernel2.csv"
# 	Es = pd.read_csv(csv_path)['Energy'].values
# 	err = pd.read_csv(csv_path)['Error'].values
# 	Es_all.append(Es)
# 	Errs.append(err)
# E_final = np.mean(Es_all[2][-10:])
# print("Final averaged energy for L=31:", E_final,"relative error:", (E_final - E_exact[2])/E_exact[2])

# fig, ax = plt.subplots(figsize=(6*0.8, 4*0.8),constrained_layout=True)

# colors = []
# for k in range(len(Ls)):
#     mantissa, exponent = f"{DHs[k]:.2e}".split("e")
#     label = rf"$L={Ls[k]},\ D_H={float(mantissa):.2f}\times 10^{{{int(exponent)}}},\ E_{{\mathrm{{exact}}}}={E_exact[k]:.3f}$"

#     inds = np.arange(0, len(Es_all[k]), 2)
#     line = ax.errorbar(
#         inds,
#         Es_all[k][inds],
#         yerr=Errs[k][inds],
#         fmt='-o',
#         capsize=2,
#         elinewidth=0.4, 
#         markersize=1,
#         linewidth=1,
#         label=label,
#     )
#     c = line[0].get_color()
#     colors.append(c)
#     ax.axhline(E_exact[k], color=c, linestyle='--')

# ax.set_xlabel(r'VMC Step')
# ax.set_ylabel(r'Energy')
# ax.legend(loc='best')
# ax.set_xlim(0, 500)
# ax.grid(False)

# # Inset: zoom into late VMC steps to show convergence
# # [x0, y0, w, h] in parent-axes fraction (0 to 1)
# # x0,y0 = lower-left corner of inset
# # w,h   = inset width/height
# axins = ax.inset_axes([0.5, 0.2, 0.45, 0.4], transform=ax.transAxes)

# # Use last 25% of steps across curves
# x_max = min(np.arange(0, len(e), 2)[-1] for e in Es_all)
# x_min = int(0.75 * x_max)

# yvals = []
# for k in range(len(Ls)):
#     inds = np.arange(0, len(Es_all[k]), 2)
#     mask = (inds >= x_min) & (inds <= x_max)
#     xk = inds[mask]
#     yk = Es_all[k][inds][mask]

#     axins.plot(xk, yk, '-o', color=colors[k], markersize=2, linewidth=1)
#     axins.axhline(E_exact[k], color=colors[k], linestyle='--', linewidth=1)

#     if len(yk) > 0:
#         yvals.extend(yk.tolist())
#     yvals.append(E_exact[k])

# axins.set_xlim(x_min, x_max)
# if len(yvals) > 0:
#     ymin, ymax = min(yvals), max(yvals)
#     pad = max(1e-5, 0.08 * (ymax - ymin))
#     axins.set_ylim(ymin - pad, ymax + pad)

# axins.tick_params(labelsize=8)
# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", lw=0.8)

# outname = '/Users/angkunwu/Desktop/vmc_exact.png'
# plt.savefig(outname, dpi=1000)
# plt.show()


#L = 13, t2=0.5, t1=1.0, J1=1.0
# J2s = np.arange(0.0, 2.1, 0.1)
# energy_exact = [-7.1901227,-7.13896884,-7.09190063,-7.04943911,-7.01222739,-6.98107004,-6.95698550,
# 				-6.94127144,-6.93557455,-6.94193570,-6.96274304,-7.00049120,-7.05729261,-7.13429925,
# 				-7.23140870,-7.34746979,-7.48077040,-7.62945057,-7.79170748,-7.96585854,-8.15035697]
# Chis = [83,79,76,77,81,81,81,82,82,75,3,88,104,114,126,134,143,145,145,150,153] # cutoff 1e-12
# Fidelity_exact = [0.85429634,0.86922009,0.88480350,0.90104827,0.91789986,0.93519768,0.95258902,0.96938821,0.98436978,
#                   0.99553181, 1.00000000-1e-10,0.99443839,0.97626674,0.94517119,0.90350522,0.85503800,0.80332282,0.75094885,
#                   0.69963718,0.65057761,0.60464161]
# energy_g0s = [-6.67400461,-6.70287845,-6.73175230,-6.76062614,-6.78949998,-6.81837382,-6.84724767,
# 			  -6.87612151,-6.90499535,-6.93386919,-6.96274304,-6.99161688,-7.02049072,-7.04936456,
# 			  -7.07823841,-7.10711225,-7.13598609,-7.16485993,-7.19373378,-7.22260762,-7.25148146]
# Fidelity_k2h16 = [0.91926730,0.91996229,0.93300474,0.94077432,0.94977760,0.95924097,0.96948266,
# 				  0.97981387,0.98925722,0.99689674,0.99992454,0.99596262,0.98026085,0.95935941,
# 				  0.92744476,0.89017510,0.84971493,0.80809152,0.76644444,0.72621667,0.68725669]
# 				#[0.84533238,0.85703021,0.88131058,0.88393199,0.90132892,0.91952372,0.93879968,0.97034514,0.97897542,
#                  # 0.98395431,0.99999613,0.99129796,0.96539015,0.92030489,0.86048895,0.80265188,0.72119343,0.64492208,
#                  # 0.60835588,0.52722341,0.47150242]
# Energy_k2h16 = [-6.84317963,-6.73161808,-6.82938094,-6.82633955,-6.83518486,-6.84385350,-6.86099528,
# 				-6.88176288,-6.90325768,-6.93379821,-6.96171269,-6.99080074,-6.97032321,-7.03861478,
# 				-7.05321950,-7.07195153,-7.08570278,-7.09749929,-7.11021666,-7.12223065,-7.09694573]
# Fidelity_k5h16 = [0.98176461,0.98364753,0.98526752,0.98591042,0.99102497,0.99328840,0.99227142,
# 				  0.99619609,0.99681216,0.99901247,0.99999851,0.99898088,0.99414760,0.98588109,
# 				  0.98684049,0.95892191,0.94360870,0.94798779,0.93801236,0.95167345,0.88379383]
# 				#[0.99329001,0.96445382,0.96489578,0.96582770,0.97432876,0.97762477,0.98422825,0.98931849,0.99403274,
#                   #0.99794006,0.99997854,0.99692392,0.98586422,0.98796952,0.98777372,0.97808599,0.92674130,0.92714155,
#                   #0.93647653,0.91792113,0.79480302]
# Energy_k5h16 = [-7.08018005,-7.04077655,-7.00833531,-6.97166447,-6.96214250,-6.93996633,-6.91809991,
# 				-6.91579423,-6.92086307,-6.93722827,-6.96272788,-6.99551251,-7.03262952,-7.07546137,
# 				-7.16092892,-7.19680205,-7.23606932,-7.31984024,-7.48264414,-7.67804192,-7.56657789]
# Fidelity_k5h64 = [0.99780947,0.99768931,0.99640191,0.99723196,0.99740493,0.99772334,0.99711430,
# 				  0.99770784,0.99886894,0.99968469,0.99999744,0.99956775,0.99851853,0.99833834,
# 				  0.99737966,0.99568617,0.99469191,0.99481761,0.99422348,0.98841441,0.99013722]
# 				#[0.99715596,0.99452996,0.99693811,0.99748856,0.99633789,0.99677116,0.99773496,0.99893361,0.99869370,
#                   #0.99965948,0.99999875,0.99929082,0.99820292,0.99865788,0.99777848,0.99333668,0.99503058,0.99408579,
#                   #0.98930866,0.99189758,0.98635614]
# Energy_k5h64 = [-7.16688696,-7.11431766,-7.06134645,-7.02521658,-6.99025724,-6.96168551,-6.93558581,
# 				-6.92661763,-6.92804576,-6.93924030,-6.96270691,-6.99659216,-7.04700743,-7.12006243,
# 				-7.20549473,-7.30925557,-7.43274075,-7.57752999,-7.70590820,-7.82210964,-8.04075694]

# energy_exact = np.array(energy_exact)
# energy_g0s = np.array(energy_g0s)
# Energy_k2h16 = np.array(Energy_k2h16)
# Energy_k5h16 = np.array(Energy_k5h16)
# Energy_k5h64 = np.array(Energy_k5h64)

# if __name__ == "__main__":
#     # fig, ax1 = plt.subplots(figsize=(6,4))

#     # # left axis: 1 - Fidelity (log scale)
#     # ax1.plot(J2s, 1 - np.array(Fidelity_exact), '-o', label=r'WF at $J_1=J_2=0$')
#     # ax1.plot(J2s, 1 - np.array(Fidelity_k2h16), '-s', label=r'kernel$=2,D_{hid}=16$')
#     # ax1.plot(J2s, 1 - np.array(Fidelity_k5h16), '-^', label=r'kernel$=5,D_{hid}=16$')
#     # ax1.plot(J2s, 1 - np.array(Fidelity_k5h64), '-x', label=r'kernel$=5,D_{hid}=64$')
#     # ax1.set_xlabel(r'$J_2$')
#     # ax1.set_ylabel(r'1 - $|\langle \Psi_{\mathrm{ed}}|\Psi_{\mathrm{test}}\rangle|$')
#     # ax1.set_yscale('log')
#     # ax1.set_ylim(1e-5, 0.6)  # Adjust y-axis limits for better visualization
#     # ax1.grid(False)
#     # ax1.legend(loc='lower left')
#     # # right axis: Chis (bond dimension)
#     # ax2 = ax1.twinx()
#     #  # black color
#     # ax2.plot(J2s, Chis, marker='o', color='black',linestyle='--',label=r'DMRG cutoff $10^{-12}$') # black color
#     # ax2.set_ylabel(r'Bond dimension $\chi$')
#     # ax2.legend(loc='lower right')
#     # ax2.set_ylim(0, 160)  
#     # plt.tight_layout()
#     # # outname = '/Users/angkunwu/Desktop/nonexactJs.png'
#     # # plt.savefig(outname, dpi=300)
#     # plt.show()

#     fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
#     	nrows=3, sharex=True, figsize=(6,6), gridspec_kw={'height_ratios': [1, 2, 2]}
# 	)

# 	# top: Bond dimension (linear)
#     ax_top.plot(J2s, Chis, marker='o', color='black', linestyle='--',
#                 label=r'DMRG cutoff $10^{-12}$')
#     ax_top.set_ylabel(r'Bond $\chi$')
#     ax_top.legend(loc='center left')
#     ax_top.set_yscale('log')
#     ax_top.set_ylim(1, 200)
#     ax_top.set_xlim(-0.02, 2.02)
#     ax_top.axhline(y=3, color='red', linestyle='--')
#     ax_top.grid(False)

# 	# middle: 1 - Fidelity (log scale)
#     y_exact = 1 - np.array(Fidelity_exact)
#     y_k2h16 = 1 - np.array(Fidelity_k2h16)
#     y_k5h16 = 1 - np.array(Fidelity_k5h16)
#     y_k5h64 = 1 - np.array(Fidelity_k5h64)

#     line1 = ax_mid.plot(J2s, y_exact, '-o', label=r'WF at $J_1=J_2=0$')
#     line2 = ax_mid.plot(J2s, y_k2h16, '-s', label=r'kernel$=2,D_\mathrm{hidden}=16$')
#     line3 = ax_mid.plot(J2s, y_k5h16, '-^', label=r'kernel$=5,D_\mathrm{hidden}=16$')
#     line4 = ax_mid.plot(J2s, y_k5h64, '-x', label=r'kernel$=5,D_\mathrm{hidden}=64$')
#     ax_mid.set_ylabel(r'1 - $\left |\langle \Psi_{\mathrm{ED}}|\Psi_{\mathrm{test}}\rangle\right|$')
#     ax_mid.set_yscale('log')
#     ax_mid.set_ylim(1e-5, 0.6)
#     ax_mid.grid(False)

#     # inset in middle: x = |J2 - 1|, each original curve -> two branches (J2<=1 and J2>=1)
#     #axins = inset_axes(ax_mid, width="46%", height="46%", loc='upper left', borderpad=1.0)
#     axins = ax_mid.inset_axes([0.63, 0.2, 0.35, 0.25]) # [x0, y0, w, h] in parent-axes fraction (0 to 1)

#     def plot_split_absJ2(ax, x, y, color, marker):
#         left = x <= 1.0
#         right = x >= 1.0

#         xl = np.abs(x[left] - 1.0)
#         yl = y[left]
#         xr = np.abs(x[right] - 1.0)
#         yr = y[right]

#         il = np.argsort(xl)
#         ir = np.argsort(xr)

#         ax.plot(xl[il], yl[il], '-', color=color, marker=marker, markersize=3, linewidth=1)
#         ax.plot(xr[ir], yr[ir], '--', color=color, marker=marker, markersize=3, linewidth=1)

#     #plot_split_absJ2(axins, J2s, y_exact, line1[0].get_color(), 'o')
#     plot_split_absJ2(axins, J2s, y_k2h16, line2[0].get_color(), 's')
#     #plot_split_absJ2(axins, J2s, y_k5h16, line3[0].get_color(), '^')
#     plot_split_absJ2(axins, J2s, y_k5h64, line4[0].get_color(), 'x')

#     axins.set_xlabel(r'$|J_2 - J_1|/J_1$', fontsize=8)
#     axins.set_yscale('log')
#     axins.set_ylim(1e-5, 0.6)
#     axins.set_xlim(-0.02, 1.02)
#     axins.tick_params(labelsize=8)
#     axins.grid(False)

#     # bottom: Energy error (log scale)
#     ax_bot.plot(J2s, (energy_g0s - energy_exact) / np.abs(energy_exact), '-o', color=line1[0].get_color(),
#             label=r'WF at $J_1=J_2=0$')
#     ax_bot.plot(J2s, (Energy_k2h16 - energy_exact) / np.abs(energy_exact), '-s', color=line2[0].get_color(),
#             label=r'kernel$=2,D_\mathrm{hidden}=16$')
#     ax_bot.plot(J2s, (Energy_k5h16 - energy_exact) / np.abs(energy_exact), '-^', color=line3[0].get_color(),
#             label=r'kernel$=5,D_\mathrm{hidden}=16$')
#     ax_bot.plot(J2s, (Energy_k5h64 - energy_exact) / np.abs(energy_exact), '-x', color=line4[0].get_color(),
#             label=r'kernel$=5,D_\mathrm{hidden}=64$')
#     ax_bot.set_ylabel(r'$(E - E_{\mathrm{exact}}) / |E_{\mathrm{exact}}|$')
#     ax_bot.set_yscale('log')
#     ax_bot.set_ylim(1e-5, 0.2)
#     ax_bot.set_xlabel(r'$J_2$')

#     # move legend from middle plot to bottom plot
#     ax_bot.legend(loc='lower right', fontsize=9)

#     plt.subplots_adjust(hspace=0.08)  # tighten vertical spacing
#     plt.tight_layout()
#     outname = '/Users/angkunwu/Desktop/nonexactJs.png'
#     plt.savefig(outname, dpi=1000)
#     plt.show()


# # Final plot system size scaling, t2=0.5, J2=0.9
# Ls = [5,7,9,11,13,15,17,19]
# # cutoff 1e-12
# Chis = [8,19,38,54,75,91,99,107]
# MPSparams = [64,298,1204,3288,7556,13344,20915,29821]
# Gaps = [1.0362295913002155,0.7292629978716909,0.5074722973133694,0.3711837290202986,
#         0.2823491878517128,0.22155926848028606,0.1782730095235543,0.1464246152814095]
# GapsExact = [1.036483828388874,0.7273239046083431,0.5070419481698378,0.37144298233760065,
#              0.28293164164842644,0.22228443045130675,0.17904748620487787,0.14719909087218674]
# # cutoff min 1- fidelity 1e-2 (10 runs), lr = 1e-3, relu, epoch 10000 or avg loss < 1e-7
# FChiddim = [8,13,26,68,190,512]
# FCparams = [249,573,1691,7821,46551,1113089] #294401]
# FChidL15 = [512,1024]
# FCparamL15 = [294401, 1113089]
# FidL15 = [0.98680770,0.99310958]
# FCparamsL15 = [249,573,1691,7821,46551,1113089]
# # Physical NN fidelity threshold 1e-3 
# Kernals = [2,2,2,4,5,6,7,8] #[2,3,3,4,5,7,9]
# Dhidden = [2,3,4,5,6,6,7,8] #[4,4,6,8,8,12,16]
# Physicalparams = [37,64,97,176,253,289,386,497] #[81,105,181,305,353,697,1153]
# PhysicalFids = [0.9999298453330994,0.9996610879898072,0.9991247057914734,0.9988309144973756,0.99885493516922,0.9986146092414856,0.9980713725090028,0.989775776863098] 
# #[0.99999667,0.99950284,0.99936825,0.99900764,0.99897051,0.99835217,0.99843025]
# RelativeEerrors = [1.67e-4,3.86e-4,5.54e-4,9.96e-4,8.85e-4,7.67e-4,9.51e-4,1.35e-2]
# # Physical NN fidelity threshold 1e-2
# # Kernals = [2,2,2,2]
# # Dhidden = [2,2,2,4]
# # Physicalparams = [37,41,45,105]
# # PhysicalFids = [0.99992979,0.99958485,0.99851441,0.99759948]
# # Physical NN relative energy error 1e-3
# Kernals2 = [2,3,4,4,5,5]
# Dhidden2 = [14,16,18,20,20,22]
# Physicalparams2 = [421,609,829,1001,1121,1321]
# PhysicalFids2 = [1.0,0.99979466,0.99906409,0.99846661,0.99561804533,0.99022579]
# PhysicalRelEnErrs2 = [7.015047e-08,2.248334e-04,9.837634e-04,8.796741e-04,4.35e-04,3.231940e-03]

# coeffsnn = np.polyfit(Ls[:len(Physicalparams)], Physicalparams, deg=2)
# coeffsmps = np.polyfit(Ls[:len(MPSparams)], MPSparams, deg=4)
# #print("Fitted polynomial coefficients:", coeffsmps)
# poly_mps = np.poly1d(coeffsmps)
# x_fit = [5,7,9,11,13,15,17,19,21]  # Extend x values for extrapolation
# y_fit_mps = poly_mps(x_fit)
# poly_nn = np.poly1d(coeffsnn)
# y_fit_nn = poly_nn(x_fit)
# print(y_fit_nn)

# if __name__ == "__main__":
#     fig, ax1 = plt.subplots(figsize=(6*0.9,4*0.9))

#     # left axis: 1 - Fidelity (log scale)
#     # line = ax1.plot(Ls[:len(Physicalparams2)], Physicalparams2, '-x', label=r'VBS NN')
#     # ax1.plot(Ls[:len(Physicalparams)], Physicalparams, '--x', label=None, color=line[0].get_color()) 
#     line1 = ax1.plot(Ls[:len(Physicalparams)], Physicalparams, 'x', label=r'VBS NN') 
#     ax1.plot(x_fit, y_fit_nn, '--', label=None, color=line1[0].get_color())
#     ax1.plot(Ls[:len(FCparams)], FCparams, '-s', label=r'FCNN')
#     #ax1.plot(Ls[:len(FCparamsL15)], FCparamsL15, 's', color='C1', linestyle='--', label=None)
#     line2 = ax1.plot(Ls[:len(MPSparams)], MPSparams, '^', label=r'MPS')
#     ax1.plot(x_fit, y_fit_mps, '--', color=line2[0].get_color(), label=None)
#     ax1.set_xlabel(r'$L$')
#     ax1.set_ylabel(r'Number of parameters')
#     ax1.set_yscale('log')
#     ax1.set_xlim(4, 20)
#     ax1.grid(False)
#     ax1.legend(loc='upper right')
#     # annotate approximate scaling near each fitted curve
#     x_nn_anno = 16
#     x_mps_anno = 14
#     ax1.text(
#         x_nn_anno,
#         poly_nn(x_nn_anno) * 1.50,   # multiplicative offset works better on log y-axis
#         rf'VBS $\sim L^{{{len(coeffsnn)-1}}}$',
#         color=line1[0].get_color(),
#         fontsize=12
#     )
#     ax1.text(
#         x_mps_anno,
#         poly_mps(x_mps_anno) * 0.5,
#         rf'MPS $\sim L^{{{len(coeffsmps)-1}}}$',
#         color=line2[0].get_color(),
#         fontsize=12
#     )
#     # right axis: gap size scaling
#     ax2 = ax1.twinx()
#      # black color
#     ax2.plot(Ls, Gaps, marker='o', color='black',linestyle='--',label=r'$\Delta$') # black color
#     ax2.set_ylabel(r'many-body gap $\Delta$')
#     ax2.legend(loc='lower center')
#     ax2.set_yscale('log')
#     #ax2.set_ylim(0, 160)  
#     plt.tight_layout()
#     outname = '/Users/angkunwu/Desktop/scalingLs.png'
#     plt.savefig(outname, dpi=1000)
#     plt.show()




# # VMC non-exact case
# Ls = [11,21,31] #[11,15,21] 
# DHs = [2772,3879876,4.81e+9] #[2772,51480,3879876]
# t2 = 0.5
# E_exact = [-2.80219475,-3.26788669,-3.66565952]#[-2.80219475, -3.00731899, -3.26788669]
# Samples = [64*128, 128*128, 128*128] # 64*128
# J1 = 0.1
# J2 = 0.09

# Es_all = []
# Errs = []
# hidds = [32,32,32]
# for L, hidden_dim in zip(Ls, hidds):
# 	#if L != 31:
# 	csv_path = f"data/energy_error_nonexact_L{L}_t2{t2}_J1{J1}_J2{J2}_hidden{hidden_dim}_kernel5.csv"
# 	#else:
# 		#csv_path = f"data/energy_error_nonexact_L{L}_t2{t2}_J1{J1}_J2{J2}_hidden{hidden_dim}_kernel5_sam16384_epoch200.csv"
# 	Es = pd.read_csv(csv_path)['Energy'].values
# 	err = pd.read_csv(csv_path)['Error'].values
# 	Es_all.append(Es)
# 	Errs.append(err)

# fig, ax = plt.subplots(figsize=(6*0.8, 4*0.8), constrained_layout=True)

# colors = []
# for k in range(len(Ls)):
#     mantissa, exponent = f"{DHs[k]:.2e}".split("e")
#     label = rf"$L={Ls[k]},\ D_H={float(mantissa):.2f}\times 10^{{{int(exponent)}}},\ E_{{\mathrm{{exact}}}}={E_exact[k]:.3f}$"

#     inds = np.arange(0, len(Es_all[k]), 1)
#     line = ax.errorbar(
#         inds,
#         Es_all[k][inds],
#         yerr=Errs[k][inds],
#         fmt='-o',
#         capsize=2,
#         elinewidth=0.4,
#         markersize=1,
#         linewidth=1,
#         label=label,
#     )
#     c = line[0].get_color()
#     colors.append(c)
#     ax.axhline(E_exact[k], color=c, linestyle='--')

# ax.set_xlabel(r'VMC Step')
# ax.set_ylabel(r'Energy')
# ax.legend(loc='best')
# ax.set_xlim(0, 200)
# ax.grid(False)

# # Inset for late-step convergence
# axins = ax.inset_axes([0.50, 0.24, 0.45, 0.40], transform=ax.transAxes)

# # Zoom to last 25% of available steps
# x_max = min(np.arange(0, len(e), 1)[-1] for e in Es_all)
# x_min = int(0.75 * x_max)

# yvals = []
# for k in range(len(Ls)):
#     inds = np.arange(0, len(Es_all[k]), 1)
#     mask = (inds >= x_min) & (inds <= x_max)
#     xk = inds[mask]
#     yk = Es_all[k][inds][mask]

#     axins.plot(xk, yk, '-o', color=colors[k], markersize=2, linewidth=1)
#     axins.axhline(E_exact[k], color=colors[k], linestyle='--', linewidth=1)

#     if len(yk) > 0:
#         yvals.extend(yk.tolist())
#     yvals.append(E_exact[k])

# axins.set_xlim(x_min, x_max)
# if len(yvals) > 0:
#     ymin, ymax = min(yvals), max(yvals)
#     pad = max(1e-5, 0.08 * (ymax - ymin))
#     axins.set_ylim(ymin - pad, ymax + pad)

# axins.tick_params(labelsize=8)
# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", lw=0.8)

# outname = '/Users/angkunwu/Desktop/vmc_nonexact.png'
# plt.savefig(outname, dpi=1000)
# plt.show()


# Plot comparison of different activations
csv_path = f"data/Loss_record_L16_J11.0_J21.0_hidden16_kernel2_activation_tanh_1.csv"
Loss_tanh_1 = pd.read_csv(csv_path)['Loss'].values
fid_tanh_1 = pd.read_csv(csv_path)['Fidelity'].values[0]
NH = pd.read_csv(csv_path)['NH'].values[0]
csv_path = f"data/Loss_record_L16_J11.0_J21.0_hidden16_kernel2_activation_tanh_3.csv"
Loss_tanh_3 = pd.read_csv(csv_path)['Loss'].values
fid_tanh_3 = pd.read_csv(csv_path)['Fidelity'].values[0]
csv_path = f"data/Loss_record_L16_J11.0_J21.0_hidden16_kernel2_activation_tanh_5.csv"
Loss_tanh_5 = pd.read_csv(csv_path)['Loss'].values
fid_tanh_5 = pd.read_csv(csv_path)['Fidelity'].values[0]
csv_path = f"data/Loss_record_L16_J11.0_J21.0_hidden16_kernel2_activation_tanh_7_2.csv"
Loss_tanh_7 = pd.read_csv(csv_path)['Loss'].values
fid_tanh_7 = pd.read_csv(csv_path)['Fidelity'].values[0]
csv_path = f"data/Loss_record_L16_J11.0_J21.0_hidden16_kernel2_activation_relu.csv"
Loss_relu = pd.read_csv(csv_path)['Loss'].values
fid_relu = pd.read_csv(csv_path)['Fidelity'].values[0]
csv_path = f"data/Loss_record_L16_J11.0_J21.0_hidden16_kernel2_activation_sigmoid.csv"
Loss_sigmoid = pd.read_csv(csv_path)['Loss'].values
fid_sigmoid = pd.read_csv(csv_path)['Fidelity'].values[0]

plt.figure(figsize=(6*0.9,5*0.9))
plt.plot(Loss_relu/NH, label=r'ReLU, $\mathrm{fidelity}$=%.7f' % fid_relu, linewidth=4)
plt.plot(Loss_sigmoid/NH, label=r'Sigmoid, $\mathrm{fidelity}$=%.7f' % fid_sigmoid)
plt.plot(Loss_tanh_1/NH, label=r'$\tanh(x)$, $\mathrm{fidelity}$=%.7f' % fid_tanh_1)
plt.plot(Loss_tanh_3/NH, label=r'$\tanh(x^3)$, $\mathrm{fidelity}$=%.7f' % fid_tanh_3)
plt.plot(Loss_tanh_5/NH, label=r'$\tanh(x^5)$, $\mathrm{fidelity}$=%.7f' % fid_tanh_5)
plt.plot(Loss_tanh_7/NH, label=r'$\tanh(x^7)$, $\mathrm{fidelity}$=%.7f' % fid_tanh_7)
plt.xlabel(r'Epoch')
plt.ylabel(r'Mean Squared Loss')
plt.xlim(-1, 2000)
plt.tick_params(axis='both', labelsize=12)
plt.legend(loc='center right', prop={'size': 9})
plt.yscale('log')
plt.grid(False)
plt.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, 1.0),   # centered, just above axes
    ncol=2,                        # 2 columns -> 3 rows for 5 entries
    prop={'size': 10}, 
    frameon=False, 
    columnspacing=1.2, 
    handlelength=1.8
)
plt.tight_layout(rect=[0, 0, 1, 0.88])  # reserve top space for legend
outname = '/Users/angkunwu/Desktop/activation_comparison.png'
plt.savefig(outname, dpi=1000)
plt.show()