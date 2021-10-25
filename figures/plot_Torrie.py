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

sigma = np.array([0.425,0.425])
Z = np.array([-1,1])
rhoplus = 0.1
c = np.array([-(Z[1]/Z[0])*rhoplus,rhoplus]) # from charge equilibrium
rhob = c*6.022e23/1.0e24 # particles/nm^3

sheetname='electrolyte11-c0.1M-sigma0.3'

df = pd.read_excel('../PyDFTele/MCdata/MCdata-Torrie1980.xls',sheet_name=sheetname) 

# print(df)

ax[0].scatter(df['z+/a'],df['rho+/rhob'],marker='o',edgecolors='C3',facecolors='none',linewidth=widthcircle,label='cations')
ax[0].scatter(df['z-/a'],df['rho-/rhob'],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='anions')

[xPB,naniPB,ncatPB,psiPB] = np.load('../profiles-PB-electrolyte11-c0.1-sigma0.3.npy')
ax[0].plot(xPB/sigma[0],naniPB/rhob[0],':',color='grey',label='PB')
ax[0].plot(xPB/sigma[0],ncatPB/rhob[1],':',color='grey')

[x,nani,ncat,psi,c1MSAani,c1MSAcat,c1nonMSAani,c1nonMSAcat] = np.load('../PyDFTele/profiles-DFTcorr-electrolyte11-c0.1-sigma0.3.npy')
ax[0].plot(x/sigma[0],nani/rhob[0],'k',label='fMSA')
ax[0].plot(x/sigma[0],ncat/rhob[1],'k')


# ax[0].set_yscale('log')
# ax[0].set_xlabel(r'$z/a_-$')
ax[0].set_ylabel(r'$\rho(z)/\rho_b$')
ax[0].set_xlim(0.0,10.0)
ax[0].set_ylim(0,10.0)
ax[0].legend(loc='upper right',ncol=2)
ax[0].tick_params(labelbottom=False)  

ax[1].scatter(df['zpsi/a'],df['Psi(kT/e)'],marker='o',edgecolors='grey',facecolors='none',linewidth=widthcircle,label='MC')

ax[1].plot(xPB/sigma[0],psiPB,':',color='grey',label='PB')

ax[1].plot(x/sigma[0],psi,'k',label='fMSA')

# ax[1].set_xlabel(r'$z/a_-$')
ax[1].set_ylabel(r'$\beta e \psi(z)$')
# ax[1].set_xlim(0.5,8.5)
ax[1].set_ylim(-1,9.0)
ax[1].text(5.0,4.5,'$Z_+ = 1$ and $Z_- = -1$')
ax[1].text(4.6,3.0,'$c = 0.1$ M and $a = 0.425$ nm')
ax[1].text(6.0,1.5,'$\sigma_* = 0.3$')
ax[1].tick_params(labelbottom=False)  

ax[2].plot(x/sigma[0],c1MSAcat,'C3',label='MSA')
ax[2].plot(x/sigma[0],c1MSAani,'C0')
ax[2].plot(x/sigma[0],c1nonMSAcat,'--',color='C3',label='nonMSA')
ax[2].plot(x/sigma[0],c1nonMSAani,'--',color='C0')

ax[2].set_xlabel(r'$z/a$')
ax[2].set_ylabel(r'$c_i^{(1)}+\mu_i$')
# ax[2].legend(loc='upper right',ncol=2)
custom_lines = [mlines.Line2D([0], [0], color='k', lw=2),
            mlines.Line2D([0], [0],ls='--', color='k', lw=2)]
ax[2].legend(custom_lines, ["MSA", "nonMSA"],loc='upper right',ncol=2)
# ax[2].set_ylim(-3,3)

fig.savefig('electrolyte11-Torrie1980-'+sheetname+'.pdf')
fig.savefig('electrolyte11-Torrie1980-'+sheetname+'.png', bbox_inches='tight')
plt.close()

#######################################################################
# figsize accepts only inches.
fig, ax = plt.subplots(3, sharex=True, sharey=False)
fig.subplots_adjust(left=0.14, right=0.96, top=0.96, bottom=0.10,
                    hspace=0.1, wspace=0.02)

sigma = np.array([0.425,0.425])
Z = np.array([-1,1])
rhoplus = 1.0
c = np.array([-(Z[1]/Z[0])*rhoplus,rhoplus]) # from charge equilibrium
rhob = c*6.022e23/1.0e24 # particles/nm^3

sheetname='electrolyte11-c1.0M-sigma0.7'

df = pd.read_excel('../PyDFTele/MCdata/MCdata-Torrie1980.xls',sheet_name=sheetname) 

# print(df)

ax[0].scatter(df['z+/a'],df['rho+/rhob'],marker='o',edgecolors='C3',facecolors='none',linewidth=widthcircle,label='cations')
ax[0].scatter(df['z-/a'],df['rho-/rhob'],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='anions')

[xPB,naniPB,ncatPB,psiPB] = np.load('../profiles-PB-electrolyte11-c1.0-sigma0.7.npy')
ax[0].plot(xPB/sigma[0],naniPB/rhob[0],':',color='grey',label='PB')
ax[0].plot(xPB/sigma[0],ncatPB/rhob[1],':',color='grey')

[x,nani,ncat,psi,c1MSAani,c1MSAcat,c1nonMSAani,c1nonMSAcat] = np.load('../PyDFTele/profiles-DFTcorr-electrolyte11-c1.0-sigma0.7.npy')
ax[0].plot(x/sigma[0],nani/rhob[0],'k',label='fMSA')
ax[0].plot(x/sigma[0],ncat/rhob[1],'k')


# ax[0].set_yscale('log')
# ax[0].set_xlabel(r'$z/a_-$')
ax[0].set_ylabel(r'$\rho(z)/\rho_b$')
ax[0].set_xlim(0.0,6.0)
ax[0].set_ylim(0,5.0)
ax[0].legend(loc='upper right',ncol=2)
ax[0].tick_params(labelbottom=False)  

ax[1].scatter(df['zpsi/a'],df['Psi(kT/e)'],marker='o',edgecolors='grey',facecolors='none',linewidth=widthcircle,label='MC')

ax[1].plot(xPB/sigma[0],psiPB,':',color='grey',label='PB')

ax[1].plot(x/sigma[0],psi,'k',label='fMSA')

# ax[1].set_xlabel(r'$z/a_-$')
ax[1].set_ylabel(r'$\beta e \psi(z)$')
# ax[1].set_xlim(0.5,8.5)
ax[1].set_ylim(-1,15.0)
ax[1].text(3.0,6.0,'$Z_+ = 1$ and $Z_- = -1$')
ax[1].text(2.6,4.0,'$c = 1.0$ M and $a = 0.425$ nm')
ax[1].text(3.6,2.0,'$\sigma_* = 0.7$')
ax[1].tick_params(labelbottom=False)  


ax[2].plot(x/sigma[0],c1MSAcat,'C3',label='MSA')
ax[2].plot(x/sigma[0],c1MSAani,'C0')
ax[2].plot(x/sigma[0],c1nonMSAcat,'--',color='C3',label='nonMSA')
ax[2].plot(x/sigma[0],c1nonMSAani,'--',color='C0')

ax[2].set_xlabel(r'$z/a$')
ax[2].set_ylabel(r'$c_i^{(1)}+\mu_i$')
# ax[2].legend(loc='upper right',ncol=2)
custom_lines = [mlines.Line2D([0], [0], color='k', lw=2),
            mlines.Line2D([0], [0],ls='--', color='k', lw=2)]
ax[2].legend(custom_lines, ["MSA", "nonMSA"],loc='upper right',ncol=2)
ax[2].set_ylim(-5,5)

fig.savefig('electrolyte11-Torrie1980-'+sheetname+'.pdf')
fig.savefig('electrolyte11-Torrie1980-'+sheetname+'.png', bbox_inches='tight')
plt.close()

#######################################################################
# figsize accepts only inches.
fig, ax = plt.subplots(3, sharex=True, sharey=False)
fig.subplots_adjust(left=0.14, right=0.96, top=0.96, bottom=0.10,
                    hspace=0.1, wspace=0.02)

sigma = np.array([0.425,0.425])
Z = np.array([-1,1])
rhoplus = 1.0
c = np.array([-(Z[1]/Z[0])*rhoplus,rhoplus]) # from charge equilibrium
rhob = c*6.022e23/1.0e24 # particles/nm^3

sheetname='electrolyte11-c1.0M-sigma0.141'

df = pd.read_excel('../PyDFTele/MCdata/MCdata-Torrie1980.xls',sheet_name=sheetname) 

# print(df)

ax[0].scatter(df['z+/a'],df['rho+/rhob'],marker='o',edgecolors='C3',facecolors='none',linewidth=widthcircle,label='cations')
ax[0].scatter(df['z-/a'],df['rho-/rhob'],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='anions')

[xPB,naniPB,ncatPB,psiPB] = np.load('../PyDFTele/profiles-PB-electrolyte11-c1.0-sigma0.141.npy')
ax[0].plot(xPB/sigma[0],naniPB/rhob[0],':',color='grey',label='PB')
ax[0].plot(xPB/sigma[0],ncatPB/rhob[1],':',color='grey')

[x,nani,ncat,psi,c1MSAani,c1MSAcat,c1nonMSAani,c1nonMSAcat] = np.load('../PyDFTele/profiles-DFTcorr-electrolyte11-c1.0-sigma0.141.npy')
ax[0].plot(x/sigma[0],nani/rhob[0],'k',label='fMSA')
ax[0].plot(x/sigma[0],ncat/rhob[1],'k')


# ax[0].set_yscale('log')
# ax[0].set_xlabel(r'$z/a_-$')
ax[0].set_ylabel(r'$\rho(z)/\rho_b$')
ax[0].set_xlim(0.0,5.0)
ax[0].set_ylim(0,6.0)
ax[0].legend(loc='upper right',ncol=2)
ax[0].tick_params(labelbottom=False)  

ax[1].scatter(df['zpsi/a'],df['Psi(kT/e)'],marker='o',edgecolors='grey',facecolors='none',linewidth=widthcircle,label='MC')

ax[1].plot(xPB/sigma[0],psiPB,':',color='grey',label='PB')

ax[1].plot(x/sigma[0],psi,'k',label='fMSA')

# ax[1].set_xlabel(r'$z/a_-$')
ax[1].set_ylabel(r'$\beta e \psi(z)$')
# ax[1].set_xlim(0.5,8.5)
ax[1].set_ylim(-1,4.0)
ax[1].text(2.5,3.1,'$Z_+ = 1$ and $Z_- = -1$')
ax[1].text(2.3,2.3,'$c = 1.0$ M and $a = 0.425$ nm')
ax[1].text(3.0,1.5,'$\sigma_* = 0.141$')
ax[1].tick_params(labelbottom=False)  

ax[2].plot(x/sigma[0],c1MSAcat,'C3',label='MSA')
ax[2].plot(x/sigma[0],c1MSAani,'C0')
ax[2].plot(x/sigma[0],c1nonMSAcat,'--',color='C3',label='nonMSA')
ax[2].plot(x/sigma[0],c1nonMSAani,'--',color='C0')

ax[2].set_xlabel(r'$z/a$')
ax[2].set_ylabel(r'$c_i^{(1)}+\mu_i$')
# ax[2].legend(loc='upper right',ncol=2)
custom_lines = [mlines.Line2D([0], [0], color='k', lw=2),
            mlines.Line2D([0], [0],ls='--', color='k', lw=2)]
ax[2].legend(custom_lines, ["MSA", "nonMSA"],loc='upper right',ncol=2)
ax[2].set_ylim(-1,1)

fig.savefig('electrolyte11-Torrie1980-'+sheetname+'.pdf')
fig.savefig('electrolyte11-Torrie1980-'+sheetname+'.png', bbox_inches='tight')
plt.close()

#######################################################################
# figsize accepts only inches.
fig, ax = plt.subplots(3, sharex=True, sharey=False)
fig.subplots_adjust(left=0.14, right=0.96, top=0.96, bottom=0.10,
                    hspace=0.1, wspace=0.02)

sigma = np.array([0.425,0.425])
Z = np.array([-1,1])
rhoplus = 2.0
c = np.array([-(Z[1]/Z[0])*rhoplus,rhoplus]) # from charge equilibrium
rhob = c*6.022e23/1.0e24 # particles/nm^3

sheetname='electrolyte11-c2.0M-sigma0.396'

df = pd.read_excel('../PyDFTele/MCdata/MCdata-Torrie1980.xls',sheet_name=sheetname) 

# print(df)

ax[0].scatter(df['z+/a'],df['rho+/rhob'],marker='o',edgecolors='C3',facecolors='none',linewidth=widthcircle,label='cations')
ax[0].scatter(df['z-/a'],df['rho-/rhob'],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='anions')

[xPB,naniPB,ncatPB,psiPB] = np.load('../profiles-PB-electrolyte11-c2.0-sigma0.396.npy')
ax[0].plot(xPB/sigma[0],naniPB/rhob[0],':',color='grey',label='PB')
ax[0].plot(xPB/sigma[0],ncatPB/rhob[1],':',color='grey')

[x,nani,ncat,psi,c1MSAani,c1MSAcat,c1nonMSAani,c1nonMSAcat] = np.load('../PyDFTele/profiles-DFTcorr-electrolyte11-c2.0-sigma0.396.npy')
ax[0].plot(x/sigma[0],nani/rhob[0],'k',label='fMSA')
ax[0].plot(x/sigma[0],ncat/rhob[1],'k')


# ax[0].set_yscale('log')
# ax[0].set_xlabel(r'$z/a_-$')
ax[0].set_ylabel(r'$\rho(z)/\rho_b$')
ax[0].set_xlim(0.0,6.0)
ax[0].set_ylim(0,4.0)
ax[0].legend(loc='upper right',ncol=2)
ax[0].tick_params(labelbottom=False)  

ax[1].scatter(df['zpsi/a'],df['Psi(kT/e)'],marker='o',edgecolors='grey',facecolors='none',linewidth=widthcircle,label='MC')

ax[1].plot(xPB/sigma[0],psiPB,':',color='grey',label='PB')

ax[1].plot(x/sigma[0],psi,'k',label='fMSA')

# ax[1].set_xlabel(r'$z/a_-$')
ax[1].set_ylabel(r'$\beta e \psi(z)$')
# ax[1].set_xlim(0.5,8.5)
ax[1].set_ylim(-1,6.0)
ax[1].text(3.0,4.0,'$Z_+ = 1$ and $Z_- = -1$')
ax[1].text(2.6,3.0,'$c = 2.0$ M and $a = 0.425$ nm')
ax[1].text(3.6,2.0,'$\sigma_* = 0.396$')
ax[1].tick_params(labelbottom=False)  


ax[2].plot(x/sigma[0],c1MSAcat,'C3',label='MSA')
ax[2].plot(x/sigma[0],c1MSAani,'C0')
ax[2].plot(x/sigma[0],c1nonMSAcat,'--',color='C3',label='nonMSA')
ax[2].plot(x/sigma[0],c1nonMSAani,'--',color='C0')

ax[2].set_xlabel(r'$z/a$')
ax[2].set_ylabel(r'$c_i^{(1)}+\mu_i$')
# ax[2].legend(loc='upper right',ncol=2)
custom_lines = [mlines.Line2D([0], [0], color='k', lw=2),
            mlines.Line2D([0], [0],ls='--', color='k', lw=2)]
ax[2].legend(custom_lines, ["MSA", "nonMSA"],loc='upper right',ncol=2)
ax[2].set_ylim(-3,3)

fig.savefig('electrolyte11-Torrie1980-'+sheetname+'.pdf')
fig.savefig('electrolyte11-Torrie1980-'+sheetname+'.png', bbox_inches='tight')
plt.close()

#######################################################################
# figsize accepts only inches.
fig, ax = plt.subplots(3, sharex=True, sharey=False)
fig.subplots_adjust(left=0.14, right=0.96, top=0.96, bottom=0.10,
                    hspace=0.1, wspace=0.02)

sigma = np.array([0.425,0.425])
Z = np.array([-1,2])
rhoplus = 0.5
c = np.array([-(Z[1]/Z[0])*rhoplus,rhoplus]) # from charge equilibrium
rhob = c*6.022e23/1.0e24 # particles/nm^3

sheetname='electrolyte21-c0.5M-sigma-0.1704'

df = pd.read_excel('../PyDFTele/MCdata/MCdata-Torrie1980.xls',sheet_name=sheetname) 

# print(df)

ax[0].scatter(df['z+/a'],df['rho+/rhob'],marker='o',edgecolors='C3',facecolors='none',linewidth=widthcircle,label='cations')
ax[0].scatter(df['z-/a'],df['rho-/rhob'],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='anions')

[xPB,naniPB,ncatPB,psiPB] = np.load('../PyDFTele/profiles-PB-electrolyte21-c0.5-sigma-0.1704.npy')
ax[0].plot(xPB/sigma[0],naniPB/rhob[0],':',color='grey',label='PB')
ax[0].plot(xPB/sigma[0],ncatPB/rhob[1],':',color='grey')

[xRFD,naniRFD,ncatRFD,psiRFD,c1MSAaniRFD,c1MSAcatRFD] = np.load('../PyDFTele/profiles-DFTRFD-electrolyte21-c0.5-sigma-0.1704.npy')
ax[0].plot(xRFD/sigma[0],naniRFD/rhob[0],'--',color='k',label='RFD')
ax[0].plot(xRFD/sigma[0],ncatRFD/rhob[1],'--',color='k')

[x,nani,ncat,psi,c1MSAani,c1MSAcat,c1nonMSAani,c1nonMSAcat] = np.load('../PyDFTele/profiles-DFTcorr-electrolyte21-c0.5-sigma-0.1704.npy')
ax[0].plot(x/sigma[0],nani/rhob[0],'-',color='k',label='fMSA')
ax[0].plot(x/sigma[0],ncat/rhob[1],'-',color='k')

# ax[0].set_yscale('log')
# ax[0].set_xlabel(r'$z/a_-$')
ax[0].set_ylabel(r'$\rho(z)/\rho_b$')
# ax[0].set_xlim(0.0,6.0)
ax[0].set_ylim(0,4.0)
ax[0].legend(loc='upper right',ncol=2)
ax[0].tick_params(labelbottom=False)  

ax[1].scatter(df['zpsi/a'],df['Psi(kT/e)'],marker='o',edgecolors='grey',facecolors='none',linewidth=widthcircle,label='MC')

ax[1].plot(xPB/sigma[0],psiPB,':',color='grey',label='PB')

ax[1].plot(xRFD/sigma[0],psiRFD,'--',color='k',label='RFD')

ax[1].plot(x/sigma[0],psi,'-',color='k',label='fMSA')

# ax[1].set_xlabel(r'$z/a_-$')
ax[1].set_ylabel(r'$\beta e \psi(z)$')
# ax[1].set_xlim(0.5,8.5)
ax[1].set_ylim(-2,0.5)
ax[1].text(2.6,-0.9,'$c_+ = 0.5$ M and $a = 0.425$ nm')
ax[1].text(3.0,-0.5,'$Z_+ = 2$ and $Z_- = -1$')
ax[1].text(3.4,-1.3,'$\sigma_* = -0.1704$')
ax[1].tick_params(labelbottom=False)  

ax[2].plot(x/sigma[0],c1MSAcat,'C3',label='MSA')
ax[2].plot(x/sigma[0],c1MSAani,'C0')
ax[2].plot(x/sigma[0],c1nonMSAcat,'--',color='C3',label='nonMSA')
ax[2].plot(x/sigma[0],c1nonMSAani,'--',color='C0')

ax[2].set_xlabel(r'$z/a$')
ax[2].set_ylabel(r'$c_i^{(1)}+\mu_i$')
custom_lines = [mlines.Line2D([0], [0], color='k', lw=2),
            mlines.Line2D([0], [0],ls='--', color='k', lw=2)]
ax[2].legend(custom_lines, ["MSA", "nonMSA"],loc='upper right',ncol=2)
ax[2].set_ylim(-2,3)

fig.savefig('electrolyte21-Torrie1980-'+sheetname+'.pdf')
fig.savefig('electrolyte21-Torrie1980-'+sheetname+'.png', bbox_inches='tight')
plt.close()

#######################################################################
# figsize accepts only inches.
fig, ax = plt.subplots(3, sharex=True, sharey=False)
fig.subplots_adjust(left=0.14, right=0.96, top=0.96, bottom=0.10,
                    hspace=0.1, wspace=0.02)

sigma = np.array([0.425,0.425])
Z = np.array([-2,2])
rhoplus = 0.5
c = np.array([-(Z[1]/Z[0])*rhoplus,rhoplus]) # from charge equilibrium
rhob = c*6.022e23/1.0e24 # particles/nm^3

sheetname='electrolyte22-c0.5M-sigma-0.1704'

df = pd.read_excel('../PyDFTele/MCdata/MCdata-Torrie1980.xls',sheet_name=sheetname) 

# print(df)

ax[0].scatter(df['z+/a'],df['rho+/rhob'],marker='o',edgecolors='C3',facecolors='none',linewidth=widthcircle,label='cations')
ax[0].scatter(df['z-/a'],df['rho-/rhob'],marker='o',edgecolors='C0',facecolors='none',linewidth=widthcircle,label='anions')

[xPB,naniPB,ncatPB,psiPB] = np.load('../profiles-PB-electrolyte22-c0.5-sigma-0.1704.npy')
ax[0].plot(xPB/sigma[0],naniPB/rhob[0],':',color='grey',label='PB')
ax[0].plot(xPB/sigma[0],ncatPB/rhob[1],':',color='grey')

[xRFD,naniRFD,ncatRFD,psiRFD,c1MSAaniRFD,c1MSAcatRFD] = np.load('../PyDFTele/profiles-DFTRFD-electrolyte21-c0.5-sigma-0.1704.npy')
ax[0].plot(xRFD/sigma[0],naniRFD/rhob[0],'--',color='k',label='RFD')
ax[0].plot(xRFD/sigma[0],ncatRFD/rhob[1],'--',color='k')

[x,nani,ncat,psi,c1MSAani,c1MSAcat,c1nonMSAani,c1nonMSAcat] = np.load('../PyDFTele/profiles-DFTcorr-electrolyte22-c0.5-sigma-0.1704.npy')
ax[0].plot(x/sigma[0],nani/rhob[0],'-',color='k',label='fMSA')
ax[0].plot(x/sigma[0],ncat/rhob[1],'-',color='k')

ax[0].set_ylabel(r'$\rho(z)/\rho_b$')
# ax[0].set_xlim(0.0,6.0)
ax[0].set_ylim(0,4.0)
ax[0].legend(loc='upper right',ncol=2)
ax[0].tick_params(labelbottom=False)  

ax[1].scatter(df['zpsi/a'],df['Psi(kT/e)'],marker='o',edgecolors='grey',facecolors='none',linewidth=widthcircle,label='MC')

ax[1].plot(xPB/sigma[0],psiPB,':',color='grey',label='PB')

ax[1].plot(xRFD/sigma[0],psiRFD,'--',color='k',label='RFD')

ax[1].plot(x/sigma[0],psi,'-',color='k',label='fMSA')

ax[1].set_ylabel(r'$\beta e \psi(z)$')
# ax[1].set_xlim(0.5,8.5)
ax[1].set_ylim(-2.5,0.5)
ax[1].text(2.6,-1.4,'$c_+ = 0.5$ M and $a = 0.425$ nm')
ax[1].text(3.0,-1,'$Z_+ =2$ and $Z_- = -2$')
ax[1].text(3.4,-1.8,'$\sigma_* = -0.1704$')
ax[1].tick_params(labelbottom=False)  

ax[2].plot(x/sigma[0],c1MSAcat,'C3',label='MSA')
ax[2].plot(x/sigma[0],c1MSAani,'C0')
ax[2].plot(x/sigma[0],c1nonMSAcat,'--',color='C3',label='nonMSA')
ax[2].plot(x/sigma[0],c1nonMSAani,'--',color='C0')

ax[2].set_xlabel(r'$z/a$')
ax[2].set_ylabel(r'$c_i^{(1)}+\mu_i$')
custom_lines = [mlines.Line2D([0], [0], color='k', lw=2),
            mlines.Line2D([0], [0],ls='--', color='k', lw=2)]
ax[2].legend(custom_lines, ["MSA", "nonMSA"],loc='upper right',ncol=2)
ax[2].set_ylim(-3,3)

fig.savefig('electrolyte22-Torrie1980-'+sheetname+'.pdf')
fig.savefig('electrolyte22-Torrie1980-'+sheetname+'.png', bbox_inches='tight')
plt.close()