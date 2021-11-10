#!/usr/bin/env python3

# This script is the python implementation of the Poisson-Boltzmann equation
#
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2021-06-02
# Updated: 2021-07-21
# Version: 1.0
#
import numpy as np
from scipy.ndimage import convolve1d

" The PB model for electrolyte solutions using the generalized grand potential"

class PBplanar():
    def __init__(self,N,delta,species=2,d=np.array([1.0,1.0]),Z=np.array([-1,1])):
        self.N = N
        self.delta = delta
        self.L = delta*N
        self.d = d
        self.species = species
        self.Z = Z
        self.nhalf = int(0.5*self.L/self.delta)

        self.lB = 0.714 # in nm (for water)
    
    def Debye_constant(self,rhob):
        return np.sqrt(4*np.pi*self.lB*np.sum(self.Z**2*rhob))

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

    def dOmegadpsi(self,rho,psi,sigma):
        lappsi = (1/(4*np.pi*self.lB))*convolve1d(psi, weights=[1,-2,1], mode='nearest')/self.delta**2
        f = np.zeros(self.N)
        f[0] = sigma[0]/self.delta
        f[-1] = sigma[1]/self.delta
        for i in range(self.species):
            f += rho[i,:]*self.Z[i]
        return lappsi + f

    def dOmegadpsi_fixedpotential(self,rho,psi,psi0):
        psi[0] = psi0[0]
        psi[-1] = psi0[1]
        lappsi = (1/(4*np.pi*self.lB))*convolve1d(psi, weights=[1,-2,1], mode='nearest')/self.delta**2
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
        d = np.array([0.3,0.3])
        delta = 0.025*min(d)
        L = 20.5*d[0]
        N = int(L/delta)
        beta = 1.0/40.0
        Z = np.array([-1,2])

        c = 0.01 #mol/L (equivalent to ionic strength for 1:1)
        M2nmunits=6.022e23/1.0e24
        rhob = np.array([-(Z[1]/Z[0])*c,c])*M2nmunits # particles/nm^3

        # fmt = FMTplanar(N,delta,species=2,d=d)
        ele = PBplanar(N,delta,species=2,d=d,Z=Z)
        x = np.linspace(0,L,N)

        # sigma = -0.1704/d[0]**2
        sigma = -3.12
        kD = np.sqrt(4*np.pi*ele.lB*np.sum(Z**2*rhob))

        n = np.ones((2,N),dtype=np.float32)
        nsig = np.array([int(0.5*d[0]/delta),int(0.5*d[1]/delta)])

        def Fpsi(psi,param):
            n[0,:nsig[0]] = 1.0e-16
            n[1,:nsig[1]] = 1.0e-16
            n[0,nsig[0]:] = param[0]*np.exp(-Z[0]*psi[nsig[0]:])
            n[1,nsig[1]:] = param[1]*np.exp(-Z[1]*psi[nsig[1]:])
            sigma = param[2]
            Fele = ele.free_energy(n,psi)
            return (Fele+sigma*psi[0])/L

        def dFpsidpsi(psi,param):
            n[0,:nsig[0]] = 1.0e-16
            n[1,:nsig[1]] = 1.0e-16
            n[0,nsig[0]:] = param[0]*np.exp(-Z[0]*psi[nsig[0]:])
            n[1,nsig[1]:] = param[1]*np.exp(-Z[1]*psi[nsig[1]:])
            sigma = param[2]
            return -ele.dOmegadpsi(n,psi,sigma)*delta/L

        param = np.array([rhob[0],rhob[1],sigma])

        psi0 = 0.1*sigma*4*np.pi*ele.lB
        psi = np.zeros(N,dtype=np.float32)
        psi[:nsig[0]] = psi0*(1/kD+0.5*d[0]-x[:nsig[0]])
        psi[nsig[0]:] = psi0*np.exp(-kD*(x[nsig[0]:]-0.5*d[0]))/kD
        # psi = psi0*np.exp(-kappa*x)
    
        [varsol,Omegasol,Niter] = optimize_fire2(psi,Fpsi,dFpsidpsi,param,1.0e-8,0.02,True)

        psi[:] = varsol

        n[0,:nsig[0]] = 1.0e-16
        n[1,:nsig[1]] = 1.0e-16
        n[0,nsig[0]:] = param[0]*np.exp(-Z[0]*psi[nsig[0]:])
        n[1,nsig[1]:] = param[1]*np.exp(-Z[1]*psi[nsig[1]:])

        # np.save('profiles-PB-electrolyte21-c0.5-d-0.1704.npy',[x,n[0],n[1],psi])
        np.save('profiles-PB-Voukadinova2018-electrolyte-Fig5-Z+=2-rho+=0.01M.npy',[x,n[0],n[1],psi])