#!/usr/bin/env python3

# This script is the python implementation of the Poisson-Boltzmann equation
#
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2021-06-02
# Updated: 2021-12-13
# Version: 1.0
#
import sys
import numpy as np
from scipy.linalg import solve_banded
from scipy.ndimage import convolve1d

def err_exit(n):
    print('Error', n, 'in tridiag.')
    sys.exit(n)

def tridiag(a, b, c, r):

    n = len(r)
    u = np.zeros(n)
    gam = np.zeros(n)
    bet = b[0]
    u[0] = r[0] / bet

    if b[0] == 0.0: err_exit(1)
    bet = b[0]
    u[0] = r[0] / bet

    # Reduce to upper triangular.
    
    for j in range(1,n):
        gam[j] = c[j-1]/bet
        bet = b[j] - a[j]*gam[j]
        if bet == 0.0: err_exit(2)
        u[j] = (r[j]-a[j]*u[j-1]) / bet

    # Solve by back-substitution.
    
    for j in range(n-2,-1,-1):
        u[j] -= gam[j+1]*u[j+1]

    return u

" The PB model for electrolyte solutions using the generalized grand potential"

class PBplanar():
    def __init__(self,N,delta,species=2,lB=0.714,d=np.array([1.0,1.0]),Z=np.array([-1,1])):
        self.N = N
        self.delta = delta
        self.L = delta*N
        self.d = d
        self.species = species
        self.Z = Z
        self.nhalf = int(0.5*self.L/self.delta)

        self.lB = lB # in nm (for water)

        # for electrostatic potential
        self.Ab = np.zeros((3,self.N))
        self.r = np.zeros(self.N)
    
    def Debye_constant(self,rhob):
        return np.sqrt(4*np.pi*self.lB*np.sum(self.Z**2*rhob))

    def Fele(self,psi):
        f = convolve1d(psi, weights=[-1,1], mode='nearest')/self.delta
        return -(1/(8*np.pi*self.lB))*np.sum(f**2)*self.delta

    def Fint(self,rho,psi):
        f =  np.sum(rho[:,:]*self.Z[:,np.newaxis],axis=0)
        return np.sum(f*psi)*self.delta

    def free_energy(self,rho,psi):
        return (self.Fele(psi)+self.Fint(rho,psi))
    
    def c1(self,psi):
        return -self.Z[:,np.newaxis]*psi[:]

    def dOmegadpsi(self,rho,psi,sigma):
        lappsi = (1/(4*np.pi*self.lB))*convolve1d(psi, weights=[1,-2,1], mode='nearest')/self.delta**2
        f = np.zeros(self.N)
        f[0] = sigma[0]/self.delta
        f[-1] = sigma[1]/self.delta
        f += np.sum(rho[:,:]*self.Z[:,np.newaxis],axis=0)
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
    test1 = False #hardwall 
    test2 = True

    import matplotlib.pyplot as plt
    from fire import optimize_fire2
    from fmt1d import FMTplanar
    from poisson1d import Poisson1D

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

    if test2: 
        d = np.array([0.3,0.3])
        delta = 0.01*min(d)
        L = 2.0
        N = int(L/delta)
        Z = np.array([-1,1])

        c = 1.0 #mol/L (equivalent to ionic strength for 1:1)
        M2nmunits=6.022e23/1.0e24
        rhob = np.array([-(Z[1]/Z[0])*c,c])*M2nmunits # particles/nm^3
        print('rhob=',rhob)

        # fmt = FMTplanar(N,delta,species=2,d=d)
        poisson = Poisson1D(N,delta,boundary_condition='mixed')
        ele = PBplanar(N,delta,species=2,d=d,Z=Z)
        x = np.linspace(0,L,N)

        # sigma = -0.1704/d[0]**2
        bound_value = np.array([-3.12,0.0]) #sigma,psi
        kD = np.sqrt(4*np.pi*ele.lB*np.sum(Z**2*rhob))

        n = np.ones((2,N),dtype=np.float32)
        nn = n.copy()
        nsig = np.array([int(0.5*d[0]/delta),int(0.5*d[1]/delta)])

        mu = np.log(rhob)
        print('mu=',rhob)

        param = np.array([rhob,bound_value])

        psi0 = bound_value[0]*4*np.pi*ele.lB/kD
        psi = np.zeros(N,dtype=np.float32)
        psi[:] = psi0*np.exp(-kD*x)

        # n[0,:] = rhob[0]*np.exp(-Z[0]*psi)
        # n[1,:] = rhob[1]*np.exp(-Z[1]*psi)

        n[0,:nsig[0]] = 1.0e-16
        n[1,:nsig[1]] = 1.0e-16
        n[0,nsig[0]:] = rhob[0]*np.exp(-Z[0]*psi[nsig[0]:])
        n[1,nsig[1]:] = rhob[1]*np.exp(-Z[1]*psi[nsig[1]:])

        # Now we will solve the DFT equations
        def Omega(var,param):
            nn[0,:] = np.exp(var[0])
            nn[1,:] = np.exp(var[1])
            Fid = np.sum(nn*(var-1.0))*delta
            Fele = ele.free_energy(nn,psi)
            bound_value = param[1,:]
            psi[:] = poisson.ElectrostaticPotential(np.sum(nn*Z[:,np.newaxis],axis=0),psi,bound_value)
            return (Fid+Fele-np.sum(mu[:,np.newaxis]*nn*delta)+bound_value[0]*psi[0])/L

        def dOmegadnR(var,param):
            nn[0,:] = np.exp(var[0])
            nn[1,:] = np.exp(var[1])

            bound_value = param[1,:]
            psi[:] = poisson.ElectrostaticPotential(np.sum(nn*Z[:,np.newaxis],axis=0),psi,bound_value)

            c1ele = ele.c1(psi)
            return nn*(var -c1ele - mu[:,np.newaxis])*delta/L        

        var = np.log(n)
        [varsol,Omegasol,Niter] = optimize_fire2(var,Omega,dOmegadnR,param,atol=1.0e-6,dt=0.2,logoutput=True)

        n[0,:nsig[0]] = 1.0e-16
        n[1,:nsig[1]] = 1.0e-16
        n[0,nsig[0]:] = rhob[0]*np.exp(-Z[0]*psi[nsig[0]:])
        n[1,nsig[1]:] = rhob[1]*np.exp(-Z[1]*psi[nsig[1]:])

        plt.plot(x,n[0])
        plt.plot(x,n[1],'C3')
        plt.show()

        plt.plot(x,psi)
        plt.show()

        # np.save('profiles-PB-electrolyte21-c0.5-d-0.1704.npy',[x,n[0],n[1],psi])
        np.save('DFTresults/profiles-PB-Voukadinova2018-electrolyte-Fig5-Z+=1-rho+=1.0M.npy',[x,n[0],n[1],psi])