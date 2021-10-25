import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.markers as mmark
import seaborn as sns
from matplotlib import cm
import pandas as pd

pts_per_inch = 72.27
text_width_in_pts = 246.0
text_width_in_inches = text_width_in_pts / pts_per_inch
ratio = 1.2 #0.618
inverse_latex_scale = 2
fig_proportion = (3.0 / 3.0)
csize = inverse_latex_scale * fig_proportion * text_width_in_inches
# always 1.0 on the first argument
fig_size = (1.0 * csize,ratio * csize)
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
          'text.latex.preamble': [r'\usepackage{amsmath}',
                                  ],
          'legend.frameon': False,
          }
plt.rcParams.update(params)
plt.ioff()
plt.clf()
# figsize accepts only inches.
fig = plt.figure(1, figsize=fig_size)
fig.subplots_adjust(left=0.14, right=0.96, top=0.96, bottom=0.14,
                    hspace=0.02, wspace=0.02)
ax = fig.add_subplot(111)

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
fig.subplots_adjust(left=0.14, right=0.96, top=0.96, bottom=0.10,
                    hspace=0.1, wspace=0.02)

sigma = np.array([0.466,0.362])
Z = np.array([-1,1])
rhoplus = 1.0
c = np.array([-(Z[1]/Z[0])*rhoplus,rhoplus]) # from charge equilibrium
rhob = c*6.022e23/1.0e24 # particles/nm^3

# sheetname='electrolyte11-c0.1M-sigma0.3'

# df = pd.read_excel('../PyDFTele/MCdata/MCdata-Torrie1980.xls',sheet_name=sheetname) 

# # print(df)

# ax[0].scatter(df['z+/a'],df['rho+/rhob'],marker='o',edgecolors='C3',facecolors='none',linewidth=widthcircle,label='cations')
# ax[0].scatter(df['z-/a'],df['rho-/rhob'],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='anions')

# [xPB,naniPB,ncatPB,psiPB] = np.load('../profiles-PB-electrolyte11-c0.1-sigma0.3.npy')
# ax[0].plot(xPB/sigma[0],naniPB/rhob[0],':',color='grey',label='PB')
# ax[0].plot(xPB/sigma[0],ncatPB/rhob[1],':',color='grey')

[xNO,naniNO,ncatNO,psiNO,c1MSAaniNO,c1MSAcatNO,c1nonMSAaniNO,c1nonMSAcatNO] = np.load('../PyDFTele/profiles-DFTfMSA-Alijo2012-electrolyte-NaI-sigma-0.1-nodispersion.npy')
ax[0].plot(xNO,naniNO/rhob[0],'--',color='k',label='fMSA')
ax[0].plot(xNO,ncatNO/rhob[1],'--',color='k')

[x,nani,ncat,psi,c1MSAani,c1MSAcat,c1nonMSAani,c1nonMSAcat] = np.load('../PyDFTele/profiles-DFTfMSA-Alijo2012-electrolyte-NaI-sigma-0.1.npy')
ax[0].plot(x,nani/rhob[0],'k',label='dispersion')
ax[0].plot(x,ncat/rhob[1],'k')

ax[0].set_ylabel(r'$\rho(z)/\rho_b$')
ax[0].set_xlim(0.0,2.0)
ax[0].set_ylim(0,6.0)
ax[0].text(0.25,5,'Na$^+$')
ax[0].text(0.3,1,'I$^-$')
ax[0].legend(loc='upper right',ncol=2)
ax[0].tick_params(labelbottom=False)  

# ax[1].scatter(df['zpsi/a'],df['Psi(kT/e)'],marker='o',edgecolors='grey',facecolors='none',linewidth=widthcircle,label='MC')

# ax[1].plot(xPB/sigma[0],psiPB,':',color='grey',label='PB')

ax[1].plot(xNO,psiNO,'--',color='k',label='fMSA')

ax[1].plot(x,psi,'k',label='dispersion')

# ax[1].set_xlabel(r'$z/a_-$')
ax[1].set_ylabel(r'$\beta e \psi(z)$')
# ax[1].set_xlim(0.5,8.5)
ax[1].set_ylim(-2.5,0.5)
ax[1].text(1.5,-0.8,'NaI')
ax[1].text(1.4,-1.3,'$I = 1.0$ M')
ax[1].text(1.3,-1.8,'$\sigma = -0.1$ C/m$^2$')
ax[1].tick_params(labelbottom=False)  

ax[2].plot(x,c1MSAcat,'C3',label='MSA')
ax[2].plot(x,c1MSAani,'C0')
ax[2].plot(x,c1nonMSAcat,'--',color='C3',label='nonMSA')
ax[2].plot(x,c1nonMSAani,'--',color='C0')

ax[2].set_xlabel(r'$z$ (nm)')
ax[2].set_ylabel(r'$c_i^{(1)}+\mu_i$')
# ax[2].legend(loc='upper right',ncol=2)
custom_lines = [mlines.Line2D([0], [0], color='k', lw=2),
            mlines.Line2D([0], [0],ls='--', color='k', lw=2)]
ax[2].legend(custom_lines, ["MSA", "nonMSA"],loc='upper right',ncol=2)
ax[2].set_ylim(-1,1)

fig.savefig('electrolyte-Alijo2012-NaI-sigma-0.1.pdf')
fig.savefig('electrolyte-Alijo2012-NaI-sigma-0.1.png', bbox_inches='tight')
plt.close()

#######################################################################
# figsize accepts only inches.
fig, ax = plt.subplots(3, sharex=True, sharey=False)
fig.subplots_adjust(left=0.14, right=0.96, top=0.96, bottom=0.10,
                    hspace=0.1, wspace=0.02)

sigma = np.array([0.732,0.362])
Z = np.array([-2,1])
rhoplus = 1.0/3.0
c = np.array([rhoplus,2*rhoplus]) # from charge equilibrium
rhob = c*6.022e23/1.0e24 # particles/nm^3

# sheetname='electrolyte11-c0.1M-sigma0.3'

# df = pd.read_excel('../PyDFTele/MCdata/MCdata-Torrie1980.xls',sheet_name=sheetname) 

# ax[0].scatter(df['z+/a'],df['rho+/rhob'],marker='o',edgecolors='C3',facecolors='none',linewidth=widthcircle,label='cations')
# ax[0].scatter(df['z-/a'],df['rho-/rhob'],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='anions')

# [xPB,naniPB,ncatPB,psiPB] = np.load('../profiles-PB-electrolyte11-c0.1-sigma0.3.npy')
# ax[0].plot(xPB/sigma[0],naniPB/rhob[0],':',color='grey',label='PB')
# ax[0].plot(xPB/sigma[0],ncatPB/rhob[1],':',color='grey')

[xNO,naniNO,ncatNO,psiNO,c1MSAaniNO,c1MSAcatNO,c1nonMSAaniNO,c1nonMSAcatNO] = np.load('../PyDFTele/profiles-DFTfMSA-Alijo2012-electrolyte-Na2SO4-sigma-0.1-nodispersion.npy')
ax[0].plot(xNO,naniNO/rhob[0],'--',color='k',label='fMSA')
ax[0].plot(xNO,ncatNO/rhob[1],'--',color='k')

[x,nani,ncat,psi,c1MSAani,c1MSAcat,c1nonMSAani,c1nonMSAcat] = np.load('../PyDFTele/profiles-DFTfMSA-Alijo2012-electrolyte-Na2SO4-sigma-0.1.npy')
ax[0].plot(x,nani/rhob[0],'k',label='dispersion')
ax[0].plot(x,ncat/rhob[1],'k')

ax[0].set_ylabel(r'$\rho(z)/\rho_b$')
ax[0].set_xlim(0.0,2.0)
ax[0].set_ylim(0,7.5)
ax[0].text(0.25,6,'Na$^+$')
ax[0].text(0.2,0.9,'SO$_4^{2-}$')
ax[0].legend(loc='upper right',ncol=2)
ax[0].tick_params(labelbottom=False)  

# ax[1].scatter(df['zpsi/a'],df['Psi(kT/e)'],marker='o',edgecolors='grey',facecolors='none',linewidth=widthcircle,label='MC')

# ax[1].plot(xPB/sigma[0],psiPB,':',color='grey',label='PB')

ax[1].plot(xNO,psiNO,'--',color='k',label='fMSA')

ax[1].plot(x,psi,'k',label='dispersion')

ax[1].set_ylabel(r'$\beta e \psi(z)$')
ax[1].set_ylim(-3,0.5)
ax[1].text(1.5,-1.5,'Na$_2$SO$_4$')
ax[1].text(1.4,-2.0,'$I = 1.0$ M')
ax[1].text(1.3,-2.5,'$\sigma = -0.1$ C/m$^2$')
ax[1].tick_params(labelbottom=False)  

ax[2].plot(x,c1MSAcat,'C3',label='MSA')
ax[2].plot(x,c1MSAani,'C0')
ax[2].plot(x,c1nonMSAcat,'--',color='C3',label='nonMSA')
ax[2].plot(x,c1nonMSAani,'--',color='C0')

ax[2].set_xlabel(r'$z$ (nm)')
ax[2].set_ylabel(r'$c_i^{(1)}+\mu_i$')
custom_lines = [mlines.Line2D([0], [0], color='k', lw=2),
            mlines.Line2D([0], [0],ls='--', color='k', lw=2)]
ax[2].legend(custom_lines, ["MSA", "nonMSA"],loc='upper right',ncol=2)
ax[2].set_ylim(-2,1)

fig.savefig('electrolyte-Alijo2012-Na2SO4-sigma-0.1.pdf')
fig.savefig('electrolyte-Alijo2012-Na2SO4-sigma-0.1.png', bbox_inches='tight')
plt.close()

#######################################################################
# figsize accepts only inches.
fig, ax = plt.subplots(3, sharex=True, sharey=False)
fig.subplots_adjust(left=0.14, right=0.96, top=0.96, bottom=0.10,
                    hspace=0.1, wspace=0.02)

sigma = np.array([0.466,0.362])
Z = np.array([-1,1])
rhoplus = 1.0
c = np.array([-(Z[1]/Z[0])*rhoplus,rhoplus]) # from charge equilibrium
rhob = c*6.022e23/1.0e24 # particles/nm^3

# sheetname='electrolyte11-c0.1M-sigma0.3'

# df = pd.read_excel('../PyDFTele/MCdata/MCdata-Torrie1980.xls',sheet_name=sheetname) 

# ax[0].scatter(df['z+/a'],df['rho+/rhob'],marker='o',edgecolors='C3',facecolors='none',linewidth=widthcircle,label='cations')
# ax[0].scatter(df['z-/a'],df['rho-/rhob'],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='anions')

# [xPB,naniPB,ncatPB,psiPB] = np.load('../profiles-PB-electrolyte11-c0.1-sigma0.3.npy')
# ax[0].plot(xPB/sigma[0],naniPB/rhob[0],':',color='grey',label='PB')
# ax[0].plot(xPB/sigma[0],ncatPB/rhob[1],':',color='grey')

[xNO,naniNO,ncatNO,psiNO,c1MSAaniNO,c1MSAcatNO,c1nonMSAaniNO,c1nonMSAcatNO] = np.load('../PyDFTele/profiles-DFTfMSA-Alijo2012-electrolyte-NaI-sigma0.1-nodispersion.npy')
ax[0].plot(xNO,naniNO/rhob[0],'--',color='k',label='fMSA')
ax[0].plot(xNO,ncatNO/rhob[1],'--',color='k')

[x,nani,ncat,psi,c1MSAani,c1MSAcat,c1nonMSAani,c1nonMSAcat] = np.load('../PyDFTele/profiles-DFTfMSA-Alijo2012-electrolyte-NaI-sigma0.1.npy')
ax[0].plot(x,nani/rhob[0],'k',label='dispersion')
ax[0].plot(x,ncat/rhob[1],'k')

ax[0].set_ylabel(r'$\rho(z)/\rho_b$')
ax[0].set_xlim(0.0,2.0)
ax[0].set_ylim(0,10.0)
ax[0].text(0.3,7.5,'I$^-$')
ax[0].text(0.3,0.8,'Na$^+$')
ax[0].legend(loc='upper right',ncol=2)
ax[0].tick_params(labelbottom=False)  

# ax[1].scatter(df['zpsi/a'],df['Psi(kT/e)'],marker='o',edgecolors='grey',facecolors='none',linewidth=widthcircle,label='MC')

# ax[1].plot(xPB/sigma[0],psiPB,':',color='grey',label='PB')

ax[1].plot(xNO,psiNO,'--',color='k',label='fMSA')

ax[1].plot(x,psi,'k',label='dispersion')

ax[1].set_ylabel(r'$\beta e \psi(z)$')
ax[1].set_ylim(-0.5,2.5)
ax[1].text(1.5,1.8,'NaI')
ax[1].text(1.4,1.3,'$I = 1.0$ M')
ax[1].text(1.35,0.8,'$\sigma = 0.1$ C/m$^2$')
ax[1].tick_params(labelbottom=False)  

ax[2].plot(x,c1MSAcat,'C3',label='MSA')
ax[2].plot(x,c1MSAani,'C0')
ax[2].plot(x,c1nonMSAcat,'--',color='C3',label='nonMSA')
ax[2].plot(x,c1nonMSAani,'--',color='C0')

ax[2].set_xlabel(r'$z$ (nm)')
ax[2].set_ylabel(r'$c_i^{(1)}+\mu_i$')
# ax[2].legend(loc='upper right',ncol=2)
custom_lines = [mlines.Line2D([0], [0], color='k', lw=2),
            mlines.Line2D([0], [0],ls='--', color='k', lw=2)]
ax[2].legend(custom_lines, ["MSA", "nonMSA"],loc='upper right',ncol=2)
ax[2].set_ylim(-1,1)

fig.savefig('electrolyte-Alijo2012-NaI-sigma0.1.pdf')
fig.savefig('electrolyte-Alijo2012-NaI-sigma0.1.png', bbox_inches='tight')
plt.close()

#######################################################################
# figsize accepts only inches.
fig, ax = plt.subplots(3, sharex=True, sharey=False)
fig.subplots_adjust(left=0.14, right=0.96, top=0.96, bottom=0.10,
                    hspace=0.1, wspace=0.02)

sigma = np.array([0.732,0.362])
Z = np.array([-2,1])
rhoplus = 1.0/3.0
c = np.array([rhoplus,2*rhoplus]) # from charge equilibrium
rhob = c*6.022e23/1.0e24 # particles/nm^3

# sheetname='electrolyte11-c0.1M-sigma0.3'

# df = pd.read_excel('../PyDFTele/MCdata/MCdata-Torrie1980.xls',sheet_name=sheetname) 

# ax[0].scatter(df['z+/a'],df['rho+/rhob'],marker='o',edgecolors='C3',facecolors='none',linewidth=widthcircle,label='cations')
# ax[0].scatter(df['z-/a'],df['rho-/rhob'],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='anions')

# [xPB,naniPB,ncatPB,psiPB] = np.load('../profiles-PB-electrolyte11-c0.1-sigma0.3.npy')
# ax[0].plot(xPB/sigma[0],naniPB/rhob[0],':',color='grey',label='PB')
# ax[0].plot(xPB/sigma[0],ncatPB/rhob[1],':',color='grey')

[xNO,naniNO,ncatNO,psiNO,c1MSAaniNO,c1MSAcatNO,c1nonMSAaniNO,c1nonMSAcatNO] = np.load('../PyDFTele/profiles-DFTfMSA-Alijo2012-electrolyte-Na2SO4-sigma0.1-nodispersion.npy')
ax[0].plot(xNO,naniNO/rhob[0],'--',color='k',label='fMSA')
ax[0].plot(xNO,ncatNO/rhob[1],'--',color='k')

[x,nani,ncat,psi,c1MSAani,c1MSAcat,c1nonMSAani,c1nonMSAcat] = np.load('../PyDFTele/profiles-DFTfMSA-Alijo2012-electrolyte-Na2SO4-sigma0.1.npy')
ax[0].plot(x,nani/rhob[0],'k',label='dispersion')
ax[0].plot(x,ncat/rhob[1],'k')

ax[0].set_ylabel(r'$\rho(z)/\rho_b$')
ax[0].set_xlim(0.0,2.0)
ax[0].set_ylim(0,45)
ax[0].text(0.15,2,'Na$^+$')
ax[0].text(0.4,38,'SO$_4^{2-}$')
ax[0].legend(loc='upper right',ncol=2)
ax[0].tick_params(labelbottom=False)  

# ax[1].scatter(df['zpsi/a'],df['Psi(kT/e)'],marker='o',edgecolors='grey',facecolors='none',linewidth=widthcircle,label='MC')

# ax[1].plot(xPB/sigma[0],psiPB,':',color='grey',label='PB')

ax[1].plot(xNO,psiNO,'--',color='k',label='fMSA')

ax[1].plot(x,psi,'k',label='dispersion')

ax[1].set_ylabel(r'$\beta e \psi(z)$')
ax[1].set_ylim(-1.5,3)
ax[1].text(1.5,2.0,'Na$_2$SO$_4$')
ax[1].text(1.45,1.5,'$I = 1.0$ M')
ax[1].text(1.4,1.0,'$\sigma = 0.1$ C/m$^2$')
ax[1].tick_params(labelbottom=False)  

ax[2].plot(x,c1MSAcat,'C3',label='MSA')
ax[2].plot(x,c1MSAani,'C0')
ax[2].plot(x,c1nonMSAcat,'--',color='C3',label='nonMSA')
ax[2].plot(x,c1nonMSAani,'--',color='C0')

ax[2].set_xlabel(r'$z$ (nm)')
ax[2].set_ylabel(r'$c_i^{(1)}+\mu_i$')
custom_lines = [mlines.Line2D([0], [0], color='k', lw=2),
            mlines.Line2D([0], [0],ls='--', color='k', lw=2)]
ax[2].legend(custom_lines, ["MSA", "nonMSA"],loc='upper right',ncol=2)
ax[2].set_ylim(-2,3)

fig.savefig('electrolyte-Alijo2012-Na2SO4-sigma0.1.pdf')
fig.savefig('electrolyte-Alijo2012-Na2SO4-sigma0.1.png', bbox_inches='tight')
plt.close()