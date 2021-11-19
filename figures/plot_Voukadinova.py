import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.markers as mmark
import seaborn as sns
from matplotlib import cm
import pandas as pd

# ###########################################################
pts_per_inch = 72.27
text_width_in_pts = 246.0
text_width_in_inches = text_width_in_pts / pts_per_inch
golden_ratio = 1.2
inverse_latex_scale = 2
fig_proportion = (3.0 / 3.0)
csize = inverse_latex_scale * fig_proportion * text_width_in_inches
# always 1.0 on the first argument
fig_size = (1.0 * csize,golden_ratio * csize)
# find out the fontsize of your latex text, and put it here
text_size = inverse_latex_scale * 9
tick_size = inverse_latex_scale * 9
legend_size = inverse_latex_scale * 8
# learn how to configure:
# http://matplotlib.sourceforge.net/users/customizing.html
params = {'backend': 'ps',
          'axes.labelsize': text_size,
          'legend.fontsize': legend_size,
          'legend.handlelength': 2.5,
          'legend.borderaxespad': 0,
          'xtick.labelsize': tick_size,
          'ytick.labelsize': tick_size,
          'font.family': 'serif',
          'font.size': text_size,
          # Times, Palatino, New Century Schoolbook,
          # Bookman, Computer Modern Roman
          'font.serif': ['Computer Modern Roman'],
          'ps.usedistiller': 'xpdf',
          'text.usetex': True,
          'figure.figsize': fig_size,
          # include here any neede package for latex
        #   'text.latex.preamble': [r'\usepackage{amsmath}',
        #                           ],
          'legend.frameon': False,
          }
plt.rcParams.update(params)
plt.ioff()
plt.clf()

# choosing the color scale
my_map = cm.get_cmap('Greens')
color = my_map(np.linspace(0.9,0.5,3,endpoint=True))

# Monte-Carlo data

widthcircle = 2
plt.rcParams["lines.markersize"] = 8.0
plt.rcParams["lines.linewidth"] = 2.0

#######################################################################
# figsize accepts only inches.
fig, ax = plt.subplots(3, sharex=True, sharey=False)
fig.subplots_adjust(left=0.16, right=0.96, top=0.96, bottom=0.10,
                    hspace=0.1, wspace=0.02)

sigma = np.array([0.3,0.15])
Z = np.array([-1,1])
rhoplus = 0.1
c = np.array([-(Z[1]/Z[0])*rhoplus,rhoplus]) # from charge equilibrium
rhob = c*6.022e23/1.0e24 # particles/nm^3

sheetname='Fig3-Z+=1-rho+=0.1M'

df = pd.read_excel('../MCdata/MCdata-Voukadinova2018.xls',sheet_name=sheetname) 

# print(df)

ax[0].scatter(df['z(nm)'],df['rho+(M)']/c[1],marker='o',edgecolors='C3',facecolors='none',linewidth=widthcircle,label='cations')
ax[0].scatter(df['z(nm)'],df['rho-(M)']/c[0],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='anions')

[xPB,naniPB,ncatPB,psiPB] = np.load('../DFTresults/profiles-PB-Voukadinova2018-electrolyte-Fig3-Z+=1-rho+=0.1M.npy')
ax[0].plot(xPB,naniPB/rhob[0],':',color='grey')
ax[0].plot(xPB,ncatPB/rhob[1],':',color='grey')


[x,nani,ncat,psi,c1MSAani,c1MSAcat] = np.load('../DFTresults/profiles-BFD-Voukadinova2018-electrolyte-Fig3-Z+=1-rho+=0.1M.npy')
ax[0].plot(x,nani/rhob[0],'k--')
ax[0].plot(x,ncat/rhob[1],'k--')


[x,nani,ncat,psi,c1MSAani,c1MSAcat,c1nonMSAani,c1nonMSAcat] = np.load('../DFTresults/profiles-DFTcorr-Voukadinova2018-electrolyte-Fig3-Z+=1-rho+=0.1M.npy')
ax[0].plot(x,nani/rhob[0],'k')
ax[0].plot(x,ncat/rhob[1],'k')

ax[0].set_yscale('log')
ax[0].set_ylabel(r'$\rho(z)/\rho_b$')
ax[0].set_xlim(0.0,4)
ax[0].set_ylim(1e-2,1e3)
ax[0].legend(loc='upper right',ncol=1)
ax[0].tick_params(labelbottom=False)  


ax[1].scatter(df['z(nm)'],df['Psi(kT/e)'],marker='o',edgecolors='grey',facecolors='none',linewidth=widthcircle,label='MC')

ax[1].plot(xPB,psiPB,':',color='grey')

[x,nani,ncat,psi,c1MSAani,c1MSAcat] = np.load('../DFTresults/profiles-BFD-Voukadinova2018-electrolyte-Fig3-Z+=1-rho+=0.1M.npy')
ax[1].plot(x,psi,'k--')

[x,nani,ncat,psi,c1MSAani,c1MSAcat,c1nonMSAani,c1nonMSAcat] = np.load('../DFTresults/profiles-DFTcorr-Voukadinova2018-electrolyte-Fig3-Z+=1-rho+=0.1M.npy')
ax[1].plot(x,psi,'k')

ax[1].set_ylabel(r'$\beta e \psi(z)$')
ax[1].set_ylim(-10,2)
ax[1].text(3,-4.5,'$c_+ = 0.1$ M', ha='center', va='center')
ax[1].text(3,-3,'$Z_+ = 1$ and $a_+ = 0.15$ nm', ha='center', va='center')
ax[1].text(3,-5.8,'$\sigma = -0.5$ C/m$^2$', ha='center', va='center')
custom_lines = [mlines.Line2D([0], [0],ls=':',color='grey', lw=2),mlines.Line2D([0], [0],ls='--',color='k', lw=2),
            mlines.Line2D([0], [0],ls='-', color='k', lw=2)]
ax[1].legend(custom_lines, ["PB","BFD","fMSA"],loc='upper center',ncol=3)
ax[1].tick_params(labelbottom=False) 

[x,nani,ncat,psi,c1MSAani,c1MSAcat] = np.load('../DFTresults/profiles-BFD-Voukadinova2018-electrolyte-Fig3-Z+=1-rho+=0.1M.npy')
ax[2].plot(x,c1MSAcat,'C3--')
ax[2].plot(x,c1MSAani,'C0--')

[x,nani,ncat,psi,c1MSAani,c1MSAcat,c1nonMSAani,c1nonMSAcat] = np.load('../DFTresults/profiles-DFTcorr-Voukadinova2018-electrolyte-Fig3-Z+=1-rho+=0.1M.npy')
ax[2].plot(x,c1MSAcat+c1nonMSAcat,'C3')
ax[2].plot(x,c1MSAani+c1nonMSAani,'C0')

ax[2].set_xlabel(r'$z$ (nm)')
ax[2].set_ylabel(r'$c^{(1),\textrm{ele-corr}}_i+\mu_{i}^\textrm{ele}$')

ax[2].set_ylim(-2,2)

fig.savefig('electrolyte-Voukadinova2018-'+sheetname+'.pdf')
fig.savefig('electrolyte-Voukadinova2018-'+sheetname+'.png', bbox_inches='tight')
plt.close()

#######################################################################
# figsize accepts only inches.
fig, ax = plt.subplots(3, sharex=True, sharey=False)
fig.subplots_adjust(left=0.16, right=0.96, top=0.96, bottom=0.10,
                    hspace=0.1, wspace=0.02)

sigma = np.array([0.3,0.15])
Z = np.array([-1,1])
rhoplus = 1.0
c = np.array([-(Z[1]/Z[0])*rhoplus,rhoplus]) # from charge equilibrium
rhob = c*6.022e23/1.0e24 # particles/nm^3

sheetname='Fig3-Z+=1-rho+=1.0M'

df = pd.read_excel('../MCdata/MCdata-Voukadinova2018.xls',sheet_name=sheetname) 

# print(df)

ax[0].scatter(df['z(nm)'],df['rho+(M)']/c[1],marker='o',edgecolors='C3',facecolors='none',linewidth=widthcircle,label='cations')
ax[0].scatter(df['z(nm)'],df['rho-(M)']/c[0],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='anions')

[xPB,naniPB,ncatPB,psiPB] = np.load('../DFTresults/profiles-PB-Voukadinova2018-electrolyte-Fig3-Z+=1-rho+=1.0M.npy')
ax[0].plot(xPB,naniPB/rhob[0],':',color='grey')
ax[0].plot(xPB,ncatPB/rhob[1],':',color='grey')

# [x,nani,ncat,psi,c1MSAani,c1MSAcat] = np.load('../DFTresults/profiles-BFD-Voukadinova2018-electrolyte-Fig3-Z+=1-rho+=1.0M.npy')
# ax[0].plot(x,nani/rhob[0],'k--')
# ax[0].plot(x,ncat/rhob[1],'k--')

[x,nani,ncat,psi,c1MSAani,c1MSAcat,c1nonMSAani,c1nonMSAcat] = np.load('../DFTresults/profiles-DFTcorr-Voukadinova2018-electrolyte-Fig3-Z+=1-rho+=1.0M.npy')
ax[0].plot(x,nani/rhob[0],'k')
ax[0].plot(x,ncat/rhob[1],'k')


ax[0].set_yscale('log')
# ax[0].set_xlabel(r'$z$ (nm)')
ax[0].set_ylabel(r'$\rho(z)/\rho_b$')
ax[0].set_xlim(0.0,2.0)
ax[0].set_ylim(1e-2,1e2)
ax[0].legend(loc='upper right',ncol=1)
ax[0].tick_params(labelbottom=False)  


ax[1].scatter(df['z(nm)'],df['Psi(kT/e)'],marker='o',edgecolors='grey',facecolors='none',linewidth=widthcircle,label='MC')

ax[1].plot(xPB,psiPB,':',color='grey')

# [x,nani,ncat,psi,c1MSAani,c1MSAcat] = np.load('../DFTresults/profiles-BFD-Voukadinova2018-electrolyte-Fig3-Z+=1-rho+=1.0M.npy')
# ax[1].plot(x,psi,'k--')

[x,nani,ncat,psi,c1MSAani,c1MSAcat,c1nonMSAani,c1nonMSAcat] = np.load('../DFTresults/profiles-DFTcorr-Voukadinova2018-electrolyte-Fig3-Z+=1-rho+=1.0M.npy')
ax[1].plot(x,psi,'k')

ax[1].set_ylabel(r'$\beta e \psi(z)$')
ax[1].set_ylim(-7,2)
ax[1].text(1.5,-4.0,'$c_+ = 1.0$ M', ha='center', va='center')
ax[1].text(1.5,-3,'$Z_+ = 1$ and $a_+ = 0.15$ nm', ha='center', va='center')
ax[1].text(1.5,-5,'$\sigma = -0.5$ C/m$^2$', ha='center', va='center')
custom_lines = [mlines.Line2D([0], [0],ls=':',color='grey', lw=2),mlines.Line2D([0], [0],ls='--',color='k', lw=2),
            mlines.Line2D([0], [0],ls='-', color='k', lw=2)]
ax[1].legend(custom_lines, ["PB","BFD","fMSA"],loc='upper center',ncol=3)
ax[1].tick_params(labelbottom=False)  

# [x,nani,ncat,psi,c1MSAani,c1MSAcat] = np.load('../DFTresults/profiles-BFD-Voukadinova2018-electrolyte-Fig3-Z+=1-rho+=1.0M.npy')
# ax[2].plot(x,c1MSAcat,'C3--')
# ax[2].plot(x,c1MSAani,'C0--')

[x,nani,ncat,psi,c1MSAani,c1MSAcat,c1nonMSAani,c1nonMSAcat] = np.load('../DFTresults/profiles-DFTcorr-Voukadinova2018-electrolyte-Fig3-Z+=1-rho+=1.0M.npy')
ax[2].plot(x,c1MSAcat+c1nonMSAcat,'C3')
ax[2].plot(x,c1MSAani+c1nonMSAani,'C0')

ax[2].set_xlabel(r'$z$ (nm)')
ax[2].set_ylabel(r'$c^{(1),\textrm{ele-corr}}_i+\mu_{i}^\textrm{ele}$')

ax[2].set_ylim(-2,2)

fig.savefig('electrolyte-Voukadinova2018-'+sheetname+'.pdf')
fig.savefig('electrolyte-Voukadinova2018-'+sheetname+'.png', bbox_inches='tight')
plt.close()

#######################################################################
# figsize accepts only inches.
fig, ax = plt.subplots(3, sharex=True, sharey=False)
fig.subplots_adjust(left=0.16, right=0.96, top=0.96, bottom=0.10,
                    hspace=0.1, wspace=0.02)

sigma = np.array([0.3,0.15])
Z = np.array([-1,2])
rhoplus = 0.1
c = np.array([-(Z[1]/Z[0])*rhoplus,rhoplus]) # from charge equilibrium
rhob = c*6.022e23/1.0e24 # particles/nm^3

sheetname='Fig3-Z+=2-rho+=0.1M'

df = pd.read_excel('../MCdata/MCdata-Voukadinova2018.xls',sheet_name=sheetname) 

# print(df)

ax[0].scatter(df['z(nm)'],df['rho+(M)']/c[1],marker='o',edgecolors='C3',facecolors='none',linewidth=widthcircle,label='cations')
ax[0].scatter(df['z(nm)'],df['rho-(M)']/c[0],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='anions')

[xPB,naniPB,ncatPB,psiPB] = np.load('../DFTresults/profiles-PB-Voukadinova2018-electrolyte-Fig3-Z+=2-rho+=0.1M.npy')
ax[0].plot(xPB,naniPB/rhob[0],':',color='grey')
ax[0].plot(xPB,ncatPB/rhob[1],':',color='grey')

[x,nani,ncat,psi,c1MSAani,c1MSAcat] = np.load('../DFTresults/profiles-BFD-Voukadinova2018-electrolyte-Fig3-Z+=2-rho+=0.1M.npy')
ax[0].plot(x,nani/rhob[0],'k--')
ax[0].plot(x,ncat/rhob[1],'k--')

[x,nani,ncat,psi,c1MSAani,c1MSAcat] = np.load('../DFTresults/profiles-fMSA-Voukadinova2018-electrolyte-Fig3-Z+=2-rho+=0.1M.npy')
ax[0].plot(x,nani/rhob[0],'k')
ax[0].plot(x,ncat/rhob[1],'k')


ax[0].set_yscale('log')
ax[0].set_ylabel(r'$\rho(z)/\rho_b$')
ax[0].set_xlim(0.0,4.0)
ax[0].set_ylim(1e-1,1e3)
ax[0].legend(loc='upper right',ncol=1)
ax[0].tick_params(labelbottom=False)  


ax[1].scatter(df['z(nm)'],df['Psi(kT/e)'],marker='o',edgecolors='grey',facecolors='none',linewidth=widthcircle,label='MC')

ax[1].plot(xPB,psiPB,':',color='grey')

[x,nani,ncat,psi,c1MSAani,c1MSAcat] = np.load('../DFTresults/profiles-BFD-Voukadinova2018-electrolyte-Fig3-Z+=2-rho+=0.1M.npy')
ax[1].plot(x,psi,'k--')

[x,nani,ncat,psi,c1MSAani,c1MSAcat] = np.load('../DFTresults/profiles-fMSA-Voukadinova2018-electrolyte-Fig3-Z+=2-rho+=0.1M.npy')
ax[1].plot(x,psi,'k')

ax[1].set_ylabel(r'$\beta e \psi(z)$')
ax[1].set_ylim(-3,1.5)
ax[1].text(3.0,-1.7,'$c_+ = 0.1$ M', ha='center', va='center')
ax[1].text(3.0,-1,'$Z_+ = 2$ and $a_+ = 0.15$ nm', ha='center', va='center')
ax[1].text(3.0,-2.4,'$\sigma = -0.5$ C/m$^2$', ha='center', va='center')
custom_lines = [mlines.Line2D([0], [0],ls=':',color='grey', lw=2),mlines.Line2D([0], [0],ls='--',color='k', lw=2),
            mlines.Line2D([0], [0],ls='-', color='k', lw=2)]
ax[1].legend(custom_lines, ["PB","BFD","fMSA"],loc='upper center',ncol=3)

[x,nani,ncat,psi,c1MSAani,c1MSAcat] = np.load('../DFTresults/profiles-BFD-Voukadinova2018-electrolyte-Fig3-Z+=2-rho+=0.1M.npy')
ax[2].plot(x,c1MSAcat,'--',color='C3')
ax[2].plot(x,c1MSAani,'--',color='C0')

[x,nani,ncat,psi,c1MSAani,c1MSAcat] = np.load('../DFTresults/profiles-fMSA-Voukadinova2018-electrolyte-Fig3-Z+=2-rho+=0.1M.npy')
ax[2].plot(x,c1MSAcat,'C3')
ax[2].plot(x,c1MSAani,'C0')

ax[2].set_xlabel(r'$z$ (nm)')
ax[2].set_ylabel(r'$c^{(1),\textrm{ele-corr}}_i+\mu_{i}^\textrm{ele}$')

ax[2].set_ylim(-3,4)

fig.savefig('electrolyte-Voukadinova2018-'+sheetname+'.pdf')
fig.savefig('electrolyte-Voukadinova2018-'+sheetname+'.png', bbox_inches='tight')
plt.close()


#######################################################################
# figsize accepts only inches.
fig, ax = plt.subplots(3, sharex=True, sharey=False)
fig.subplots_adjust(left=0.16, right=0.96, top=0.96, bottom=0.10,
                    hspace=0.1, wspace=0.02)

sigma = np.array([0.3,0.15])
Z = np.array([-1,2])
rhoplus = 1.0
c = np.array([-(Z[1]/Z[0])*rhoplus,rhoplus]) # from charge equilibrium
rhob = c*6.022e23/1.0e24 # particles/nm^3

sheetname='Fig3-Z+=2-rho+=1.0M'

df = pd.read_excel('../MCdata/MCdata-Voukadinova2018.xls',sheet_name=sheetname) 

# print(df)

ax[0].scatter(df['z(nm)'],df['rho+(M)']/c[1],marker='o',edgecolors='C3',facecolors='none',linewidth=widthcircle,label='cations')
ax[0].scatter(df['z(nm)'],df['rho-(M)']/c[0],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='anions')

[xPB,naniPB,ncatPB,psiPB] = np.load('../DFTresults/profiles-PB-Voukadinova2018-electrolyte-Fig3-Z+=2-rho+=1.0M.npy')
ax[0].plot(xPB,naniPB/rhob[0],':',color='grey')
ax[0].plot(xPB,ncatPB/rhob[1],':',color='grey')

[x,nani,ncat,psi,c1MSAani,c1MSAcat] = np.load('../DFTresults/profiles-BFD-Voukadinova2018-electrolyte-Fig3-Z+=2-rho+=1.0M.npy')
ax[0].plot(x,nani/rhob[0],'--',color='k')
ax[0].plot(x,ncat/rhob[1],'--',color='k')

[x,nani,ncat,psi,c1MSAani,c1MSAcat,c1nonMSAani,c1nonMSAcat] = np.load('../DFTresults/profiles-fMSA-Voukadinova2018-electrolyte-Fig3-Z+=2-rho+=1.0M.npy')
ax[0].plot(x,nani/rhob[0],'k')
ax[0].plot(x,ncat/rhob[1],'k')

ax[0].set_yscale('log')
# ax[0].set_xlabel(r'$z$ (nm)')
ax[0].set_ylabel(r'$\rho(z)/\rho_b$')
ax[0].set_xlim(0.0,2.0)
ax[0].set_ylim(1e-1,1e2)
ax[0].legend(loc='upper right',ncol=1)
ax[0].tick_params(labelbottom=False)  


ax[1].scatter(df['z(nm)'],df['Psi(kT/e)'],marker='o',edgecolors='grey',facecolors='none',linewidth=widthcircle,label='MC')

ax[1].plot(xPB,psiPB,':',color='grey')

[x,nani,ncat,psi,c1MSAani,c1MSAcat] = np.load('../DFTresults/profiles-BFD-Voukadinova2018-electrolyte-Fig3-Z+=2-rho+=1.0M.npy')
ax[1].plot(x,psi,'--',color='k')

[x,nani,ncat,psi,c1MSAani,c1MSAcat,c1nonMSAani,c1nonMSAcat] = np.load('../DFTresults/profiles-fMSA-Voukadinova2018-electrolyte-Fig3-Z+=2-rho+=1.0M.npy')
ax[1].plot(x,psi,'-',color='k')

# ax[1].set_xlabel(r'$z$ (nm)')
ax[1].set_ylabel(r'$\beta e \psi(z)$')
# ax[1].set_xlim(0.5,8.5)
ax[1].set_ylim(-3,2)
ax[1].text(1.5,-1.7,'$c_+ = 1.0$ M', ha='center', va='center')
ax[1].text(1.5,-1,'$Z_+ = 2$ and $a_+ = 0.15$ nm', ha='center', va='center')
ax[1].text(1.5,-2.4,'$\sigma = -0.5$ C/m$^2$', ha='center', va='center')
custom_lines = [mlines.Line2D([0], [0],ls=':',color='grey', lw=2),mlines.Line2D([0], [0],ls='--',color='k', lw=2),
            mlines.Line2D([0], [0],ls='-', color='k', lw=2)]
ax[1].legend(custom_lines, ["PB","BFD","fMSA"],loc='upper center',ncol=3)
ax[1].tick_params(labelbottom=False)  

[x,nani,ncat,psi,c1MSAani,c1MSAcat] = np.load('../DFTresults/profiles-BFD-Voukadinova2018-electrolyte-Fig3-Z+=2-rho+=1.0M.npy')
ax[2].plot(x,c1MSAcat,'--',color='C3')
ax[2].plot(x,c1MSAani,'--',color='C0')

[x,nani,ncat,psi,c1MSAani,c1MSAcat,c1nonMSAani,c1nonMSAcat] = np.load('../DFTresults/profiles-fMSA-Voukadinova2018-electrolyte-Fig3-Z+=2-rho+=1.0M.npy')
ax[2].plot(x,c1MSAcat+c1nonMSAcat,'C3')
ax[2].plot(x,c1MSAani+c1nonMSAani,'C0')

ax[2].set_xlabel(r'$z$ (nm)')
ax[2].set_ylabel(r'$c^{(1),\textrm{ele-corr}}_i+\mu_{i}^\textrm{ele}$')
ax[2].set_ylim(-7,7)

fig.savefig('electrolyte-Voukadinova2018-'+sheetname+'.pdf')
fig.savefig('electrolyte-Voukadinova2018-'+sheetname+'.png', bbox_inches='tight')
plt.close()

#######################################################################
# figsize accepts only inches.
fig, ax = plt.subplots(3, sharex=True, sharey=False)
fig.subplots_adjust(left=0.16, right=0.96, top=0.96, bottom=0.10,
                    hspace=0.1, wspace=0.02)

sigma = np.array([0.3,0.15])
Z = np.array([-1,3])
rhoplus = 0.1
c = np.array([-(Z[1]/Z[0])*rhoplus,rhoplus]) # from charge equilibrium
rhob = c*6.022e23/1.0e24 # particles/nm^3

sheetname='Fig3-Z+=3-rho+=0.1M'

df = pd.read_excel('../MCdata/MCdata-Voukadinova2018.xls',sheet_name=sheetname) 

# print(df)

ax[0].scatter(df['z(nm)'],df['rho+(M)']/c[1],marker='o',edgecolors='C3',facecolors='none',linewidth=widthcircle,label='cations')
ax[0].scatter(df['z(nm)'],df['rho-(M)']/c[0],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='anions')

[xPB,naniPB,ncatPB,psiPB] = np.load('../DFTresults/profiles-PB-Voukadinova2018-electrolyte-Fig3-Z+=3-rho+=0.1M.npy')
ax[0].plot(xPB,naniPB/rhob[0],':',color='grey')
ax[0].plot(xPB,ncatPB/rhob[1],':',color='grey')

[x,nani,ncat,psi,c1MSAani,c1MSAcat] = np.load('../DFTresults/profiles-BFD-Voukadinova2018-electrolyte-Fig3-Z+=3-rho+=0.1M.npy')
ax[0].plot(xPB,naniPB/rhob[0],'--',color='k')
ax[0].plot(xPB,ncatPB/rhob[1],'--',color='k')

[x,nani,ncat,psi,c1MSAani,c1MSAcat] = np.load('../DFTresults/profiles-fMSA-Voukadinova2018-electrolyte-Fig3-Z+=3-rho+=0.1M.npy')
ax[0].plot(x,nani/rhob[0],'k')
ax[0].plot(x,ncat/rhob[1],'k')


ax[0].set_yscale('log')
# ax[0].set_xlabel(r'$z$ (nm)')
ax[0].set_ylabel(r'$\rho(z)/\rho_b$')
ax[0].set_xlim(0.0,4)
ax[0].set_ylim(1e-1,1e3)
ax[0].legend(loc='upper right',ncol=1)
ax[0].tick_params(labelbottom=False)  


ax[1].scatter(df['z(nm)'],df['Psi(kT/e)'],marker='o',edgecolors='grey',facecolors='none',linewidth=widthcircle,label='MC')

ax[1].plot(xPB,psiPB,':',color='grey')

[x,nani,ncat,psi,c1MSAani,c1MSAcat] = np.load('../DFTresults/profiles-BFD-Voukadinova2018-electrolyte-Fig3-Z+=3-rho+=0.1M.npy')
ax[1].plot(x,psi,'k--')

[x,nani,ncat,psi,c1MSAani,c1MSAcat] = np.load('../DFTresults/profiles-fMSA-Voukadinova2018-electrolyte-Fig3-Z+=3-rho+=0.1M.npy')
ax[1].plot(x,psi,'k')

# ax[1].set_xlabel(r'$z$ (nm)')
ax[1].set_ylabel(r'$\beta e \psi(z)$')
# ax[1].set_xlim(0.5,8.5)
ax[1].set_ylim(-1,3)
ax[1].text(2.5,1.4,'$c_+ = 0.1$ M', ha='center', va='center')
ax[1].text(2.5,1.9,'$Z_+ = 3$ and $a_+ = 0.15$ nm', ha='center', va='center')
ax[1].text(2.5,0.9,'$\sigma = -0.5$ C/m$^2$', ha='center', va='center')
custom_lines = [mlines.Line2D([0], [0],ls=':',color='grey', lw=2),mlines.Line2D([0], [0],ls='--',color='k', lw=2),
            mlines.Line2D([0], [0],ls='-', color='k', lw=2)]
ax[1].legend(custom_lines, ["PB","BFD","fMSA"],loc='upper center',ncol=3)
ax[1].tick_params(labelbottom=False)  

[x,nani,ncat,psi,c1MSAani,c1MSAcat] = np.load('../DFTresults/profiles-BFD-Voukadinova2018-electrolyte-Fig3-Z+=3-rho+=0.1M.npy')
ax[2].plot(x,c1MSAcat,'C3--')
ax[2].plot(x,c1MSAani,'C0--')

[x,nani,ncat,psi,c1MSAani,c1MSAcat] = np.load('../DFTresults/profiles-fMSA-Voukadinova2018-electrolyte-Fig3-Z+=3-rho+=0.1M.npy')
ax[2].plot(x,c1MSAcat,'C3')
ax[2].plot(x,c1MSAani,'C0')

ax[2].set_xlabel(r'$z$ (nm)')
ax[2].set_ylabel(r'$c^{(1),\textrm{ele-corr}}_i+\mu_{i}^\textrm{ele}$')

# ax[2].set_ylim(-5,10)

fig.savefig('electrolyte-Voukadinova2018-'+sheetname+'.pdf')
fig.savefig('electrolyte-Voukadinova2018-'+sheetname+'.png', bbox_inches='tight')
plt.close()


#######################################################################
# figsize accepts only inches.
fig, ax = plt.subplots(3, sharex=True, sharey=False)
fig.subplots_adjust(left=0.16, right=0.96, top=0.96, bottom=0.10,
                    hspace=0.1, wspace=0.02)

sigma = np.array([0.3,0.15])
Z = np.array([-1,3])
rhoplus = 1.0
c = np.array([-(Z[1]/Z[0])*rhoplus,rhoplus]) # from charge equilibrium
rhob = c*6.022e23/1.0e24 # particles/nm^3

sheetname='Fig3-Z+=3-rho+=1.0M'

df = pd.read_excel('../MCdata/MCdata-Voukadinova2018.xls',sheet_name=sheetname) 

# print(df)

ax[0].scatter(df['z(nm)'],df['rho+(M)']/c[1],marker='o',edgecolors='C3',facecolors='none',linewidth=widthcircle,label='cations')
ax[0].scatter(df['z(nm)'],df['rho-(M)']/c[0],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='anions')

[xPB,naniPB,ncatPB,psiPB] = np.load('../DFTresults/profiles-PB-Voukadinova2018-electrolyte-Fig3-Z+=3-rho+=1.0M.npy')
ax[0].plot(xPB,naniPB/rhob[0],':',color='grey')
ax[0].plot(xPB,ncatPB/rhob[1],':',color='grey')

[x,nani,ncat,psi,c1MSAani,c1MSAcat] = np.load('../DFTresults/profiles-BFD-Voukadinova2018-electrolyte-Fig3-Z+=3-rho+=1.0M.npy')
ax[0].plot(x,nani/rhob[0],'k--')
ax[0].plot(x,ncat/rhob[1],'k--')

[x,nani,ncat,psi,c1MSAani,c1MSAcat,c1nonMSAani,c1nonMSAcat] = np.load('../DFTresults/profiles-DFTcorr-Voukadinova2018-electrolyte-Fig3-Z+=3-rho+=1.0M.npy')
ax[0].plot(x,nani/rhob[0],'k')
ax[0].plot(x,ncat/rhob[1],'k')

ax[0].set_yscale('log')
# ax[0].set_xlabel(r'$z$ (nm)')
ax[0].set_ylabel(r'$\rho(z)/\rho_b$')
ax[0].set_xlim(0.0,2.0)
ax[0].set_ylim(1e-1,1e2)
ax[0].legend(loc='upper right',ncol=1)
ax[0].tick_params(labelbottom=False)  


ax[1].scatter(df['z(nm)'],df['Psi(kT/e)'],marker='o',edgecolors='grey',facecolors='none',linewidth=widthcircle,label='MC')

ax[1].plot(xPB,psiPB,':',color='grey')

[x,nani,ncat,psi,c1MSAani,c1MSAcat] = np.load('../DFTresults/profiles-BFD-Voukadinova2018-electrolyte-Fig3-Z+=3-rho+=1.0M.npy')
ax[1].plot(x,psi,'k--')

[x,nani,ncat,psi,c1MSAani,c1MSAcat,c1nonMSAani,c1nonMSAcat] = np.load('../DFTresults/profiles-DFTcorr-Voukadinova2018-electrolyte-Fig3-Z+=3-rho+=1.0M.npy')
ax[1].plot(x,psi,'k')


ax[1].set_ylabel(r'$\beta e \psi(z)$')
ax[1].set_ylim(-3,3)
ax[1].text(1.5,-1.7,'$c_+ = 1.0$ M', ha='center', va='center')
ax[1].text(1.5,-1.0,'$Z_+ = 3$ and $a_+ = 0.15$ nm', ha='center', va='center')
ax[1].text(1.5,-2.4,'$\sigma = -0.5$ C/m$^2$', ha='center', va='center')
custom_lines = [mlines.Line2D([0], [0],ls=':',color='grey', lw=2),mlines.Line2D([0], [0],ls='--',color='k', lw=2),
            mlines.Line2D([0], [0],ls='-', color='k', lw=2)]
ax[1].legend(custom_lines, ["PB","BFD","fMSA"],loc='upper center',ncol=3)
ax[1].tick_params(labelbottom=False)  

[x,nani,ncat,psi,c1MSAani,c1MSAcat] = np.load('../DFTresults/profiles-BFD-Voukadinova2018-electrolyte-Fig3-Z+=3-rho+=1.0M.npy')
ax[2].plot(x,c1MSAcat,'C3--')
ax[2].plot(x,c1MSAani,'C0--')

[x,nani,ncat,psi,c1MSAani,c1MSAcat,c1nonMSAani,c1nonMSAcat] = np.load('../DFTresults/profiles-DFTcorr-Voukadinova2018-electrolyte-Fig3-Z+=3-rho+=1.0M.npy')
ax[2].plot(x,c1MSAcat+c1nonMSAcat,'C3')
ax[2].plot(x,c1MSAani+c1nonMSAani,'C0')

ax[2].set_xlabel(r'$z$ (nm)')
ax[2].set_ylabel(r'$c^{(1),\textrm{ele-corr}}_i+\mu_{i}^\textrm{ele}$')

ax[2].set_ylim(-4,12)

fig.savefig('electrolyte-Voukadinova2018-'+sheetname+'.pdf')
fig.savefig('electrolyte-Voukadinova2018-'+sheetname+'.png', bbox_inches='tight')
plt.close()

#######################################################################
# figsize accepts only inches.
fig, ax = plt.subplots(3, sharex=True, sharey=False)
fig.subplots_adjust(left=0.16, right=0.96, top=0.96, bottom=0.10,
                    hspace=0.1, wspace=0.02)

sigma = np.array([0.3,0.3])
Z = np.array([-1,1])
c = np.array([0.01,0.01])
rhob = c*6.022e23/1.0e24 # particles/nm^3

df = pd.read_excel('../MCdata/MCdata-Voukadinova2018.xls',sheet_name='Fig5-Z+=1-rho+=0.01M') 

# print(df)

ax[0].scatter(df['z(nm)'],df['rho+(M)']/c[1],marker='o',edgecolors='C3',facecolors='none',linewidth=widthcircle,label='cations')
ax[0].scatter(df['z(nm)'],df['rho-(M)']/c[0],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='anions')

[xPB,naniPB,ncatPB,psiPB] = np.load('../DFTresults/profiles-PB-Voukadinova2018-electrolyte-Fig5-Z+=1-rho+=0.01M.npy')
ax[0].plot(xPB,naniPB/rhob[0],':',color='grey')
ax[0].plot(xPB,ncatPB/rhob[1],':',color='grey')

[x,nani,ncat,psi,c1MSAani,c1MSAcat] = np.load('../DFTresults/profiles-BFD-Voukadinova2018-electrolyte-Fig5-Z+=1-rho+=0.01M.npy')
ax[0].plot(x,nani/rhob[0],'k--')
ax[0].plot(x,ncat/rhob[1],'k--')

[x,nani,ncat,psi,c1MSAani,c1MSAcat,c1nonMSAani,c1nonMSAcat] = np.load('../DFTresults/profiles-DFTcorr-Voukadinova2018-electrolyte-Fig5-Z+=1-rho+=0.01M.npy')
ax[0].plot(x,nani/rhob[0],'k')
ax[0].plot(x,ncat/rhob[1],'k')

ax[0].set_yscale('log')
ax[0].set_ylabel(r'$\rho(z)/\rho_b$')
ax[0].set_xlim(0.0,12)
ax[0].set_ylim(1e-3,1e4)
ax[0].legend(loc='upper right',ncol=1)
ax[0].tick_params(labelbottom=False)  


ax[1].scatter(df['z(nm)'],df['Psi(kT/e)'],marker='o',edgecolors='grey',facecolors='none',linewidth=widthcircle,label='MC')

ax[1].plot(xPB,psiPB,':',color='grey')

[x,nani,ncat,psi,c1MSAani,c1MSAcat] = np.load('../DFTresults/profiles-BFD-Voukadinova2018-electrolyte-Fig5-Z+=1-rho+=0.01M.npy')
ax[1].plot(x,psi,'k--')

[x,nani,ncat,psi,c1MSAani,c1MSAcat,c1nonMSAani,c1nonMSAcat] = np.load('../DFTresults/profiles-DFTcorr-Voukadinova2018-electrolyte-Fig5-Z+=1-rho+=0.01M.npy')
ax[1].plot(x,psi,'k')

ax[1].set_ylabel(r'$\beta e \psi(z)$')
ax[1].set_ylim(-15,5)
ax[1].text(8.5,-9,'$c_+ = 0.01$ M', ha='center', va='center')
ax[1].text(8.5,-7,'$Z_+ = 1$ and $a_+ = 0.3$ nm', ha='center', va='center')
ax[1].text(8.5,-11,'$\sigma = -0.5$ C/m$^2$', ha='center', va='center')
custom_lines = [mlines.Line2D([0], [0],ls=':',color='grey', lw=2),mlines.Line2D([0], [0],ls='--',color='k', lw=2),
            mlines.Line2D([0], [0],ls='-', color='k', lw=2)]
ax[1].legend(custom_lines, ["PB","BFD","fMSA"],loc='upper center',ncol=3)
ax[1].tick_params(labelbottom=False)  

[x,nani,ncat,psi,c1MSAani,c1MSAcat] = np.load('../DFTresults/profiles-BFD-Voukadinova2018-electrolyte-Fig5-Z+=1-rho+=0.01M.npy')
ax[2].plot(x,c1MSAcat,'C3--')
ax[2].plot(x,c1MSAani,'C0--')

[x,nani,ncat,psi,c1MSAani,c1MSAcat,c1nonMSAani,c1nonMSAcat] = np.load('../DFTresults/profiles-DFTcorr-Voukadinova2018-electrolyte-Fig5-Z+=1-rho+=0.01M.npy')
ax[2].plot(x,c1MSAcat+c1nonMSAcat,'C3')
ax[2].plot(x,c1MSAani+c1nonMSAani,'C0')


ax[2].set_xlabel(r'$z$ (nm)')
ax[2].set_ylabel(r'$c^{(1),\textrm{ele-corr}}_i+\mu_{i}^\textrm{ele}$')

ax[2].set_ylim(-4,4)

fig.savefig('electrolyte-Voukadinova2018-Fig5-Z+=1-rho+=0.01M.pdf')
fig.savefig('electrolyte-Voukadinova2018-Fig5-Z+=1-rho+=0.01M.png', bbox_inches='tight')
plt.close()


#######################################################################
# figsize accepts only inches.
fig, ax = plt.subplots(3, sharex=True, sharey=False)
fig.subplots_adjust(left=0.16, right=0.96, top=0.96, bottom=0.10,
                    hspace=0.1, wspace=0.02)

sigma = np.array([0.3,0.3])
Z = np.array([-1,1])
rhoplus = 1.0
c = np.array([-(Z[1]/Z[0])*rhoplus,rhoplus]) # from charge equilibrium
rhob = c*6.022e23/1.0e24 # particles/nm^3

sheetname='Fig5-Z+=1-rho+=1.0M'

df = pd.read_excel('../MCdata/MCdata-Voukadinova2018.xls',sheet_name=sheetname) 

# print(df)

ax[0].scatter(df['z(nm)'],df['rho+(M)']/c[1],marker='o',edgecolors='C3',facecolors='none',linewidth=widthcircle,label='cations')
ax[0].scatter(df['z(nm)'],df['rho-(M)']/c[0],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='anions')

[xPB,naniPB,ncatPB,psiPB] = np.load('../DFTresults/profiles-PB-Voukadinova2018-electrolyte11-Fig5-Z+=1-rho+=1.0M.npy')
ax[0].plot(xPB,naniPB/rhob[0],':',color='grey')
ax[0].plot(xPB,ncatPB/rhob[1],':',color='grey')

# [x,nani,ncat,psi,c1MSAani,c1MSAcat] = np.load('../DFTresults/profiles-BFD-Voukadinova2018-electrolyte-Fig5-Z+=1-rho+=1.0M.npy')
# ax[0].plot(x,nani/rhob[0],'k--')
# ax[0].plot(x,ncat/rhob[1],'k--')

[x,nani,ncat,psi,c1MSAani,c1MSAcat,c1nonMSAani,c1nonMSAcat] = np.load('../DFTresults/profiles-DFTcorr-Voukadinova2018-electrolyte-Fig5-Z+=1-rho+=1.0M.npy')
ax[0].plot(x,nani/rhob[0],'k')
ax[0].plot(x,ncat/rhob[1],'k')

ax[0].set_yscale('log')
# ax[0].set_xlabel(r'$z$ (nm)')
ax[0].set_ylabel(r'$\rho(z)/\rho_b$')
ax[0].set_xlim(0.0,2)
ax[0].set_ylim(1e-3,1e2)
ax[0].legend(loc='upper right',ncol=1)
ax[0].tick_params(labelbottom=False)  


ax[1].scatter(df['z(nm)'],df['Psi(kT/e)'],marker='o',edgecolors='grey',facecolors='none',linewidth=widthcircle,label='MC')

ax[1].plot(xPB,psiPB,':',color='grey')
ax[1].plot(x,psi,'k')

ax[1].set_ylabel(r'$\beta e \psi(z)$')
ax[1].set_ylim(-8,2.5)
ax[1].text(1.5,-4.5,'$c_+ = 1.0$ M', ha='center', va='center')
ax[1].text(1.5,-3,'$Z_+ = 1$ and $a_+ = 0.3$ nm', ha='center', va='center')
ax[1].text(1.5,-6,'$\sigma = -0.5$ C/m$^2$', ha='center', va='center')
custom_lines = [mlines.Line2D([0], [0],ls=':',color='grey', lw=2),mlines.Line2D([0], [0],ls='--',color='k', lw=2),
            mlines.Line2D([0], [0],ls='-', color='k', lw=2)]
ax[1].legend(custom_lines, ["PB","BFD","fMSA"],loc='upper center',ncol=3)
ax[1].tick_params(labelbottom=False)  


ax[2].plot(x,c1MSAcat+c1nonMSAcat,'C3')
ax[2].plot(x,c1MSAani+c1nonMSAani,'C0')


ax[2].set_xlabel(r'$z$ (nm)')
ax[2].set_ylabel(r'$c^{(1),\textrm{ele-corr}}_i+\mu_{i}^\textrm{ele}$')

# ax[2].set_ylim(-3,3)

fig.savefig('electrolyte-Voukadinova2018-'+sheetname+'.pdf')
fig.savefig('electrolyte-Voukadinova2018-'+sheetname+'.png', bbox_inches='tight')
plt.close()



#######################################################################
# figsize accepts only inches.
fig, ax = plt.subplots(3, sharex=True, sharey=False)
fig.subplots_adjust(left=0.16, right=0.96, top=0.96, bottom=0.10,
                    hspace=0.1, wspace=0.02)

sigma = np.array([0.3,0.3])
Z = np.array([-1,2])
rhoplus = 0.01
c = np.array([-(Z[1]/Z[0])*rhoplus,rhoplus]) # from charge equilibrium
rhob = c*6.022e23/1.0e24 # particles/nm^3

sheetname='Fig5-Z+=2-rho+=0.01M'

df = pd.read_excel('../MCdata/MCdata-Voukadinova2018.xls',sheet_name=sheetname) 

ax[0].scatter(df['z(nm)'],df['rho+(M)']/c[1],marker='o',edgecolors='C3',facecolors='none',linewidth=widthcircle,label='cations')
ax[0].scatter(df['z(nm)'],df['rho-(M)']/c[0],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='anions')

[xPB,naniPB,ncatPB,psiPB] = np.load('../DFTresults/profiles-PB-Voukadinova2018-electrolyte-Fig5-Z+=2-rho+=0.01M.npy')
ax[0].plot(xPB,naniPB/rhob[0],':',color='grey')
ax[0].plot(xPB,ncatPB/rhob[1],':',color='grey')

[x,nani,ncat,psi,c1MSAani,c1MSAcat,c1nonMSAani,c1nonMSAcat] = np.load('../DFTresults/profiles-DFTcorr-Voukadinova2018-electrolyte-Fig5-Z+=2-rho+=0.01M.npy')
ax[0].plot(x,nani/rhob[0],'k')
ax[0].plot(x,ncat/rhob[1],'k')


ax[0].set_yscale('log')
ax[0].set_ylabel(r'$\rho(z)/\rho_b$')
ax[0].set_xlim(0.0,6)
ax[0].set_ylim(1e-2,1e4)
ax[0].legend(loc='upper right',ncol=1)
ax[0].tick_params(labelbottom=False)  


ax[1].scatter(df['z(nm)'],df['Psi(kT/e)'],marker='o',edgecolors='grey',facecolors='none',linewidth=widthcircle,label='MC')

ax[1].plot(xPB,psiPB,':',color='grey')

ax[1].plot(x,psi,'k')

ax[1].set_ylabel(r'$\beta e \psi(z)$')
ax[1].set_ylim(-6,1)
ax[1].text(4,-4.0,'$c_+ = 0.01$ M', ha='center', va='center')
ax[1].text(4,-3,'$Z_+ = 2$ and $a_+ = 0.3$ nm', ha='center', va='center')
ax[1].text(4,-5,'$\sigma = -0.5$ C/m$^2$', ha='center', va='center')
custom_lines = [mlines.Line2D([0], [0],ls=':',color='grey', lw=2),mlines.Line2D([0], [0],ls='--',color='k', lw=2),
            mlines.Line2D([0], [0],ls='-', color='k', lw=2)]
ax[1].legend(custom_lines, ["PB","BFD","fMSA"],loc='upper center',ncol=3)
ax[1].tick_params(labelbottom=False)  

ax[2].plot(x,c1MSAcat+c1nonMSAcat,'C3')
ax[2].plot(x,c1MSAani+c1nonMSAani,'C0')


ax[2].set_xlabel(r'$z$ (nm)')
ax[2].set_ylabel(r'$c^{(1),\textrm{ele-corr}}_i+\mu_{i}^\textrm{ele}$')

# ax[1].set_xlim(0.5,8.5)
# ax[1].set_ylim(-3,3)

fig.savefig('electrolyte-Voukadinova2018-'+sheetname+'.pdf')
fig.savefig('electrolyte-Voukadinova2018-'+sheetname+'.png', bbox_inches='tight')
plt.close()

#######################################################################
# figsize accepts only inches.
fig, ax = plt.subplots(3, sharex=True, sharey=False)
fig.subplots_adjust(left=0.16, right=0.96, top=0.96, bottom=0.10,
                    hspace=0.1, wspace=0.02)

sigma = np.array([0.3,0.3])
Z = np.array([-1,2])
rhoplus = 1.0
c = np.array([-(Z[1]/Z[0])*rhoplus,rhoplus]) # from charge equilibrium
rhob = c*6.022e23/1.0e24 # particles/nm^3

sheetname='Fig5-Z+=2-rho+=1.0M'

df = pd.read_excel('../MCdata/MCdata-Voukadinova2018.xls',sheet_name=sheetname) 

# print(df)

ax[0].scatter(df['z(nm)'],df['rho+(M)']/c[1],marker='o',edgecolors='C3',facecolors='none',linewidth=widthcircle,label='cations')
ax[0].scatter(df['z(nm)'],df['rho-(M)']/c[0],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='anions')

[xPB,naniPB,ncatPB,psiPB] = np.load('../DFTresults/profiles-PB-Voukadinova2018-electrolyte-Fig5-Z+=2-rho+=1.0M.npy')
ax[0].plot(xPB,naniPB/rhob[0],':',color='grey')
ax[0].plot(xPB,ncatPB/rhob[1],':',color='grey')

[x,nani,ncat,psi,c1MSAani,c1MSAcat,c1nonMSAani,c1nonMSAcat] = np.load('../DFTresults/profiles-DFTcorr-Voukadinova2018-electrolyte-Fig5-Z+=2-rho+=1.0M.npy')
ax[0].plot(x,nani/rhob[0],'k')
ax[0].plot(x,ncat/rhob[1],'k')


ax[0].set_yscale('log')
ax[0].set_ylabel(r'$\rho(z)/\rho_b$')
ax[0].set_xlim(0.0,2)
ax[0].set_ylim(1e-1,1e2)
ax[0].legend(loc='upper right',ncol=1)
ax[0].tick_params(labelbottom=False)  


ax[1].scatter(df['z(nm)'],df['Psi(kT/e)'],marker='o',edgecolors='grey',facecolors='none',linewidth=widthcircle,label='MC')

ax[1].plot(xPB,psiPB,':',color='grey')

ax[1].plot(x,psi,'k')

ax[1].set_ylabel(r'$\beta e \psi(z)$')
ax[1].set_ylim(-4.5,2)
ax[1].text(1.5,-2,'$c_+ = 1.0$ M', ha='center', va='center')
ax[1].text(1.5,-1,'$Z_+ = 2$ and $a_+ = 0.3$ nm', ha='center', va='center')
ax[1].text(1.5,-3,'$\sigma = -0.5$ C/m$^2$', ha='center', va='center')
custom_lines = [mlines.Line2D([0], [0],ls=':',color='grey', lw=2),mlines.Line2D([0], [0],ls='--',color='k', lw=2),
            mlines.Line2D([0], [0],ls='-', color='k', lw=2)]
ax[1].legend(custom_lines, ["PB","BFD","fMSA"],loc='upper center',ncol=3)
ax[1].tick_params(labelbottom=False)  

ax[2].plot(x,c1MSAcat+c1nonMSAcat,'C3')
ax[2].plot(x,c1MSAani+c1nonMSAani,'C0')

ax[2].set_xlabel(r'$z$ (nm)')
ax[2].set_ylabel(r'$c^{(1),\textrm{ele-corr}}_i+\mu_{i}^\textrm{ele}$')

# ax[1].set_xlim(0.5,8.5)
ax[2].set_ylim(-5,10)


fig.savefig('electrolyte-Voukadinova2018-'+sheetname+'.pdf')
fig.savefig('electrolyte-Voukadinova2018-'+sheetname+'.png', bbox_inches='tight')
plt.close()



#######################################################################
# figsize accepts only inches.
fig, ax = plt.subplots(3, sharex=True, sharey=False)
fig.subplots_adjust(left=0.16, right=0.96, top=0.96, bottom=0.10,
                    hspace=0.1, wspace=0.02)

sigma = np.array([0.3,0.3])
Z = np.array([-1,3])
rhoplus = 0.01
c = np.array([-(Z[1]/Z[0])*rhoplus,rhoplus]) # from charge equilibrium
rhob = c*6.022e23/1.0e24 # particles/nm^3

sheetname='Fig5-Z+=3-rho+=0.01M'

df = pd.read_excel('../MCdata/MCdata-Voukadinova2018.xls',sheet_name=sheetname) 

# print(df)

ax[0].scatter(df['z(nm)'],df['rho+(M)']/c[1],marker='o',edgecolors='C3',facecolors='none',linewidth=widthcircle,label='cations')
ax[0].scatter(df['z(nm)'],df['rho-(M)']/c[0],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='anions')

[xPB,naniPB,ncatPB,psiPB] = np.load('../DFTresults/profiles-PB-Voukadinova2018-electrolyte-Fig5-Z+=3-rho+=0.01M.npy')
ax[0].plot(xPB,naniPB/rhob[0],':',color='grey')
ax[0].plot(xPB,ncatPB/rhob[1],':',color='grey')

[x,nani,ncat,psi,c1MSAani,c1MSAcat,c1nonMSAani,c1nonMSAcat] = np.load('../DFTresults/profiles-DFTcorr-Voukadinova2018-electrolyte-Fig5-Z+=3-rho+=0.01M.npy')
ax[0].plot(x,nani/rhob[0],'k')
ax[0].plot(x,ncat/rhob[1],'k')


ax[0].set_yscale('log')
ax[0].set_ylabel(r'$\rho(z)/\rho_b$')
ax[0].set_xlim(0.0,6)
ax[0].set_ylim(1e-2,1e4)
ax[0].legend(loc='upper right',ncol=1)
ax[0].tick_params(labelbottom=False)  


ax[1].scatter(df['z(nm)'],df['Psi(kT/e)'],marker='o',edgecolors='grey',facecolors='none',linewidth=widthcircle,label='MC')

ax[1].plot(xPB,psiPB,':',color='grey')
ax[1].plot(x,psi,'k')

ax[1].set_ylabel(r'$\beta e \psi(z)$')
ax[1].set_ylim(-3,3)
ax[1].text(4.5,-1.8,'$c_+ = 0.01$ M', ha='center', va='center')
ax[1].text(4.5,-1.0,'$Z_+ = 3$ and $a_+ = 0.3$ nm', ha='center', va='center')
ax[1].text(4.5,-2.6,'$\sigma = -0.5$ C/m$^2$', ha='center', va='center')
custom_lines = [mlines.Line2D([0], [0],ls=':',color='grey', lw=2),mlines.Line2D([0], [0],ls='--',color='k', lw=2),
            mlines.Line2D([0], [0],ls='-', color='k', lw=2)]
ax[1].legend(custom_lines, ["PB","BFD","fMSA"],loc='upper center',ncol=3)
ax[1].tick_params(labelbottom=False)  

ax[2].plot(x,c1MSAcat+c1nonMSAcat,'C3')
ax[2].plot(x,c1MSAani+c1nonMSAani,'C0')

ax[2].set_xlabel(r'$z$ (nm)')
ax[2].set_ylabel(r'$c^{(1),\textrm{ele-corr}}_i+\mu_{i}^\textrm{ele}$')

# ax[1].set_xlim(0.5,8.5)
# ax[1].set_ylim(-3,3)

fig.savefig('electrolyte-Voukadinova2018-'+sheetname+'.pdf')
fig.savefig('electrolyte-Voukadinova2018-'+sheetname+'.png', bbox_inches='tight')
plt.close()

#######################################################################
# figsize accepts only inches.
fig, ax = plt.subplots(3, sharex=True, sharey=False)
fig.subplots_adjust(left=0.16, right=0.96, top=0.96, bottom=0.10,
                    hspace=0.1, wspace=0.02)

sigma = np.array([0.3,0.3])
Z = np.array([-1,3])
rhoplus = 1.0
c = np.array([-(Z[1]/Z[0])*rhoplus,rhoplus]) # from charge equilibrium
rhob = c*6.022e23/1.0e24 # particles/nm^3

sheetname='Fig5-Z+=3-rho+=1.0M'

df = pd.read_excel('../MCdata/MCdata-Voukadinova2018.xls',sheet_name=sheetname) 

ax[0].scatter(df['z(nm)'],df['rho+(M)']/c[1],marker='o',edgecolors='C3',facecolors='none',linewidth=widthcircle,label='cations')
ax[0].scatter(df['z(nm)'],df['rho-(M)']/c[0],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='anions')

[xPB,naniPB,ncatPB,psiPB] = np.load('../DFTresults/profiles-PB-Voukadinova2018-electrolyte-Fig5-Z+=3-rho+=1.0M.npy')
ax[0].plot(xPB,naniPB/rhob[0],':',color='grey')
ax[0].plot(xPB,ncatPB/rhob[1],':',color='grey')

[x,nani,ncat,psi,c1MSAani,c1MSAcat,c1nonMSAani,c1nonMSAcat] = np.load('../DFTresults/profiles-DFTcorr-Voukadinova2018-electrolyte-Fig5-Z+=3-rho+=1.0M.npy')
ax[0].plot(x,nani/rhob[0],'k')
ax[0].plot(x,ncat/rhob[1],'k')

ax[0].set_yscale('log')
ax[0].set_ylabel(r'$\rho(z)/\rho_b$')
ax[0].set_xlim(0.0,2)
ax[0].set_ylim(1e-1,1e2)
ax[0].legend(loc='upper right',ncol=1)
ax[0].tick_params(labelbottom=False)  


ax[1].scatter(df['z(nm)'],df['Psi(kT/e)'],marker='o',edgecolors='grey',facecolors='none',linewidth=widthcircle,label='MC')

ax[1].plot(xPB,psiPB,':',color='grey')

ax[1].plot(x,psi,'k')

ax[1].set_ylabel(r'$\beta e \psi(z)$')
ax[1].set_ylim(-3,4)
ax[1].text(1.5,2,'$c_+ = 1.0$ M', ha='center', va='center')
ax[1].text(1.5,3,'$Z_+ = 3$ and $a_+ = 0.3$ nm', ha='center', va='center')
ax[1].text(1.5,1,'$\sigma = -0.5$ C/m$^2$', ha='center', va='center')
custom_lines = [mlines.Line2D([0], [0],ls=':',color='grey', lw=2),mlines.Line2D([0], [0],ls='--',color='k', lw=2),
            mlines.Line2D([0], [0],ls='-', color='k', lw=2)]
ax[1].legend(custom_lines, ["PB","BFD","fMSA"],loc='upper center',ncol=3)
ax[1].tick_params(labelbottom=False)  

ax[2].plot(x,c1MSAcat+c1nonMSAcat,'C3')
ax[2].plot(x,c1MSAani+c1nonMSAani,'C0')

ax[2].set_xlabel(r'$z$ (nm)')
ax[2].set_ylabel(r'$c^{(1),\textrm{ele-corr}}_i+\mu_{i}^\textrm{ele}$')

ax[2].set_ylim(-5,10)

fig.savefig('electrolyte-Voukadinova2018-'+sheetname+'.pdf')
fig.savefig('electrolyte-Voukadinova2018-'+sheetname+'.png', bbox_inches='tight')
plt.close()
