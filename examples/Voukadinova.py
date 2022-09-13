import numpy as np
import sys
sys.path.insert(0, '../src/')
from electrolyte import ElectrolyteDFT1D
import matplotlib.pyplot as plt
import pandas as pd


plt.style.use(['science'])

d = np.array([0.3,0.3])
L = 3.0
Z = np.array([-1,1])

c = 1.0 #mol/L (equivalent to ionic strength for 1:1)
rhob = np.array([-(Z[1]/Z[0])*c,c])*6.022e23/1.0e24 # particles/nm^3

ele = ElectrolyteDFT1D(L=L,d=d,Z=Z,ecmethod='BFD')
ele.Set_BulkDensities(rhob)
ele.Set_External_Potential(extpotmodel='hardwall')

# sigma = -0.1704/d[0]**2
sigma = -3.125 

ele.Set_Boundary_Conditions(params=np.array([sigma,0.0]),bc='sigma')
# ele.Set_Boundary_Conditions(params=np.array([-10,0.0]),bc='potential')
ele.Set_InitialCondition()
ele.Update_System()

# plt.plot(ele.z,ele.rho[0]/rhob[0])
# plt.plot(ele.z,ele.rho[1]/rhob[1],'C3')
# plt.show()

# plt.plot(ele.z,ele.psi)
# plt.show()

ele.Calculate_Equilibrium(method='fire',dt=0.002,rtol=1e-3,atol=1e-4,logoutput=True)

sheetname='Fig5-Z+=1-rho+='+str(c)+'M'
df = pd.read_excel('MCdata/MCdata-Voukadinova2018.xls',sheet_name=sheetname) 

cion = np.array([-(Z[1]/Z[0])*c,c]) # from charge equilibrium
plt.scatter(df['z(nm)'],df['rho+(M)']/cion[1],marker='o',edgecolors='C3',facecolors='none',label='cations')
plt.scatter(df['z(nm)'],df['rho-(M)']/cion[0],marker='o',edgecolors='C0',facecolors='none',label='anions')

plt.yscale('log')
plt.plot(ele.z,ele.rho[0]/rhob[0],'k',label='BFD')
plt.plot(ele.z,ele.rho[1]/rhob[1],'k')
plt.xlabel(r'$z$ (nm)')
plt.ylabel(r'$\rho_i(z)/\rho_b$')
plt.xlim(0.0,2)
plt.ylim(1e-3,1e2)
plt.text(1.2,1e-1,'$c_+ = $'+str(c)+' M', ha='center', va='center')
plt.text(1.2,4e-2,'$Z_+ = 1$ and $a_+ = 0.3$ nm', ha='center', va='center')
plt.text(1.2,2e-2,'$\sigma = -0.5$ C/m$^2$', ha='center', va='center')
plt.legend(loc='upper right',ncol=1)
plt.savefig('ionprofile-electrolyte-Voukadinova2018-'+sheetname+'.png',dpi=200)
plt.show()

plt.plot(ele.z,ele.psi,'k',label='BFD')
plt.xlim(0.0,2)
plt.ylim(top=0)
plt.xlabel(r'$z$ (nm)')
plt.ylabel(r'$\beta e \psi(z)$')
plt.text(1.2,-5.5,'$c_+ = $'+str(c)+' M', ha='center', va='center')
plt.text(1.2,-6,'$Z_+ = 1$ and $a_+ = 0.3$ nm', ha='center', va='center')
plt.text(1.2,-6.5,'$\sigma = -0.5$ C/m$^2$', ha='center', va='center')
plt.savefig('potential-electrolyte-Voukadinova2018-'+sheetname+'.png',dpi=200)
plt.show()

