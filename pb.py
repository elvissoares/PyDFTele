import numpy as np
from scipy.ndimage import convolve1d
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2020-11-18
# Updated: 2021-06-14

" The DFT model for electrolyte solutions using the generalized grand potential"

class PB():
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
            f += rho[i,:]*Z[i]
        return np.sum(f*psi)*self.delta

    def free_energy(self,rho,psi,beta):
        return (self.Fele(psi)+self.Fint(rho,psi))
    
    def c1(self,psi):
        cc = np.empty((self.species,self.N))
        for i in range(self.species):
            cc[i,:] = -Z[i]*psi[i,:]
        return cc

    def dOmegadpsi(self,rho,psi,Gamma):
        lappsi = (1/(4*np.pi*self.lB))*convolve1d(psi, weights=[1,-2,1], mode='nearest')/self.delta**2
        lappsi[0] += Gamma/self.delta
        # lappsi[-1] += Gamma/self.delta
        f = np.zeros(self.N)
        for i in range(self.species):
            f += rho[i,:]*Z[i]
        return lappsi + f

if __name__ == "__main__":
    test1 = True #hardwall 

    import matplotlib.pyplot as plt
    import sys
    # sys.path.append("/home/elvis/Documents/Projetos em Andamento/DFT models/PyDFT/")
    from fire import optimize_fire2
    from fmt import FMTplanar

    if test1: 
        sigma = np.array([0.425,0.425])
        delta = 0.02*min(sigma)
        N = 500
        L = N*delta
        beta = 1.0/40.0
        Z = np.array([-2,2])

        c = 0.5 #mol/L (equivalent to ionic strength for 1:1)
        M2nmunits=6.022e23/1.0e24
        rhob = np.array([c,c])*M2nmunits # particles/nm^3

        # fmt = FMTplanar(N,delta,species=2,sigma=sigma)
        ele = PB(N,delta,species=2,sigma=sigma,Z=Z)
        x = np.linspace(0,L,N)

        Gamma = -0.1704/sigma[0]**2
        kappa = np.sqrt(4*np.pi*ele.lB*np.sum(Z**2*rhob))

        n = np.ones((2,N),dtype=np.float32)
        nsig = np.array([int(0.5*sigma[0]/delta),int(0.5*sigma[1]/delta)])

        def Fpsi(psi,param):
            n[0,:nsig[0]] = 1.0e-16
            n[1,:nsig[1]] = 1.0e-16
            n[0,nsig[0]:] = param[0]*np.exp(-Z[0]*psi[nsig[0]:])
            n[1,nsig[1]:] = param[1]*np.exp(-Z[1]*psi[nsig[1]:])
            Gamma = param[2]
            Fele = ele.free_energy(n,psi,beta)
            return (Fele+Gamma*psi[0])/L

        def dFpsidpsi(psi,param):
            n[0,:nsig[0]] = 1.0e-16
            n[1,:nsig[1]] = 1.0e-16
            n[0,nsig[0]:] = param[0]*np.exp(-Z[0]*psi[nsig[0]:])
            n[1,nsig[1]:] = param[1]*np.exp(-Z[1]*psi[nsig[1]:])
            Gamma = param[2]
            return -ele.dOmegadpsi(n,psi,Gamma)*delta/L

        param = np.array([rhob[0],rhob[1],Gamma])

        psi0 = Gamma*4*np.pi*ele.lB
        psi = psi0*np.exp(-kappa*x)
    
        [varsol,Omegasol,Niter] = optimize_fire2(psi,Fpsi,dFpsidpsi,param,1.0e-8,0.02,True)

        psi[:] = varsol

        n[0,:nsig[0]] = 1.0e-16
        n[1,:nsig[1]] = 1.0e-16
        n[0,nsig[0]:] = param[0]*np.exp(-Z[0]*psi[nsig[0]:])
        n[1,nsig[1]:] = param[1]*np.exp(-Z[1]*psi[nsig[1]:])

        np.save('profiles-PB-electrolyte22-c0.5-sigma-0.1704.npy',[x,n[0],n[1],psi])