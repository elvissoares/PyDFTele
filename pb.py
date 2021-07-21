import numpy as np
from scipy.ndimage import convolve1d
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2020-11-18
# Updated: 2021-06-14

" The DFT model for electrolyte solutions using the generalized grand potential"

class PBplanar():
    def __init__(self,N,delta,species=2,sigma=np.array([1.0,1.0]),Z=np.array([-1,1])):
        self.N = N
        self.delta = delta
        self.L = delta*N
        self.sigma = sigma
        self.species = species
        self.Z = Z

        self.lB = 0.714 # in nm (for water)

    def Fele(self,psi):
        f = convolve1d(psi, weights=[-1,1], mode='nearest')/self.delta
        return -(1/(8*np.pi*self.lB))*np.sum(f**2)*self.delta

    def Fint(self,rho,psi):
        f = np.zeros(self.N)
        for i in range(self.species):
            f += rho[i,:]*self.Z[i]
        return np.sum(f*psi)*self.delta

    def free_energy(self,rho,psi):
        return (self.Fele(psi)+self.Fint(rho,psi))
    
    def c1(self,psi):
        cc = np.empty((self.species,self.N))
        for i in range(self.species):
            cc[i,:] = -self.Z[i]*psi[i,:]
        return cc

    def dOmegadpsi(self,rho,psi,Gamma):
        lappsi = (1/(4*np.pi*self.lB))*convolve1d(psi, weights=[1,-2,1], mode='nearest')/self.delta**2
        lappsi[0] += Gamma/self.delta
        # lappsi[-1] += Gamma/self.delta
        f = np.zeros(self.N)
        for i in range(self.species):
            f += rho[i,:]*self.Z[i]
        return lappsi + f

if __name__ == "__main__":
    test1 = True #hardwall 

    import matplotlib.pyplot as plt
    from fire import optimize_fire2
    from fmt import FMTplanar

    if test1: 
        sigma = np.array([0.3,0.3])
        delta = 0.025*min(sigma)
        L = 45*sigma[0]
        N = int(L/delta)
        beta = 1.0/40.0
        Z = np.array([-1,1])

        c = 0.01 #mol/L (equivalent to ionic strength for 1:1)
        M2nmunits=6.022e23/1.0e24
        rhob = np.array([-(Z[1]/Z[0])*c,c])*M2nmunits # particles/nm^3

        # fmt = FMTplanar(N,delta,species=2,sigma=sigma)
        ele = PBplanar(N,delta,species=2,sigma=sigma,Z=Z)
        x = np.linspace(0,L,N)

        # Gamma = -0.1704/sigma[0]**2
        Gamma = -3.12
        kD = np.sqrt(4*np.pi*ele.lB*np.sum(Z**2*rhob))

        n = np.ones((2,N),dtype=np.float32)
        nsig = np.array([int(0.5*sigma[0]/delta),int(0.5*sigma[1]/delta)])

        def Fpsi(psi,param):
            n[0,:nsig[0]] = 1.0e-16
            n[1,:nsig[1]] = 1.0e-16
            n[0,nsig[0]:] = param[0]*np.exp(-Z[0]*psi[nsig[0]:])
            n[1,nsig[1]:] = param[1]*np.exp(-Z[1]*psi[nsig[1]:])
            Gamma = param[2]
            Fele = ele.free_energy(n,psi)
            return (Fele+Gamma*psi[0])/L

        def dFpsidpsi(psi,param):
            n[0,:nsig[0]] = 1.0e-16
            n[1,:nsig[1]] = 1.0e-16
            n[0,nsig[0]:] = param[0]*np.exp(-Z[0]*psi[nsig[0]:])
            n[1,nsig[1]:] = param[1]*np.exp(-Z[1]*psi[nsig[1]:])
            Gamma = param[2]
            return -ele.dOmegadpsi(n,psi,Gamma)*delta/L

        param = np.array([rhob[0],rhob[1],Gamma])

        psi0 = 0.1*Gamma*4*np.pi*ele.lB
        psi = np.zeros(N,dtype=np.float32)
        psi[:nsig[0]] = psi0*(1/kD+0.5*sigma[0]-x[:nsig[0]])
        psi[nsig[0]:] = psi0*np.exp(-kD*(x[nsig[0]:]-0.5*sigma[0]))/kD
        # psi = psi0*np.exp(-kappa*x)
    
        [varsol,Omegasol,Niter] = optimize_fire2(psi,Fpsi,dFpsidpsi,param,1.0e-8,0.02,True)

        psi[:] = varsol

        n[0,:nsig[0]] = 1.0e-16
        n[1,:nsig[1]] = 1.0e-16
        n[0,nsig[0]:] = param[0]*np.exp(-Z[0]*psi[nsig[0]:])
        n[1,nsig[1]:] = param[1]*np.exp(-Z[1]*psi[nsig[1]:])

        # np.save('profiles-PBplanar-Voukadinova2018-electrolyte-c0.5-sigma-0.1704.npy',[x,n[0],n[1],psi])
        np.save('profiles-PBplanar-Voukadinova2018-electrolyte-Fig5-Z+=1-rho+=0.01M.npy',[x,n[0],n[1],psi])