#!/usr/bin/env python3

# This script is the python implementation of the Density Functional Theory
# for Electrolyte Solution in the presence of an external electrostatic potential
#
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2021-06-02
# Updated: 2021-07-20
# Version: 1.0
#
import numpy as np
from scipy.ndimage import convolve1d
from scipy import optimize
import matplotlib.pyplot as plt
from numba import jit

def phicorr1D(x,b,a):
    ba = b/a
    conds = [(np.abs(x)<=a), (np.abs(x)>a)]
    funcs = [lambda x: (2*np.pi*a**3/3)*(1-3*ba+3*ba**2-3*ba**2*np.abs(x/a)+3*ba*np.abs(x/a)**2-np.abs(x/a)**3),0.0]
    return np.piecewise(x,conds,funcs)

# The MSA parameters
# @jit(nopython=True)
def Hfunc(rho,a,Gamma):
    return np.sum(rho*a[:,np.newaxis]**3/(1+Gamma*a[:,np.newaxis]),axis=0)+(2.0/np.pi)*(1-(np.pi/6)*np.sum(rho*a[:,np.newaxis]**3,axis=0))

# @jit(nopython=True, parallel=True)
def Etafunc(rho,a,Z,Gamma):
    return np.sum(rho*Z[:,np.newaxis]*a[:,np.newaxis]/(1+Gamma*a[:,np.newaxis]),axis=0)/Hfunc(rho,a,Gamma)

# @jit(nopython=True, parallel=True)
def Gammafunc(rho,a,Z,lB):
    def fun(x):
        eta = Etafunc(rho,a,Z,x)
        return x - np.sqrt(np.pi*lB*np.sum(rho*((Z[:,np.newaxis]-eta*a[:,np.newaxis]**2)/(1+x*a[:,np.newaxis]))**2,axis=0))
    kappa = np.sqrt(4*np.pi*lB*np.sum(Z[:,np.newaxis]**2*rho,axis=0))
    amed = np.power(np.sum(rho*a[:,np.newaxis]**3,axis=0)/np.sum(rho,axis=0),1.0/3.0)
    x0 = (np.sqrt(1+2*kappa*amed)-1)/(2*amed)
    sol = optimize.root(fun, x0, jac=None, method='hybr')
    return sol.x

def Gammafuncbulk(rho,a,Z,lB):
    def fun(x):
        Hh = np.sum(rho*a**3/(1+x*a))+(2.0/np.pi)*(1-(np.pi/6)*np.sum(rho*a**3))
        eta = np.sum(rho*Z*a/(1+x*a))/Hh
        return x - np.sqrt(np.pi*lB*np.sum(rho*((Z-eta*a**2)/(1+x*a))**2))
    kappa = np.sqrt(4*np.pi*lB*np.sum(Z**2*rho))
    amed = np.power(np.sum(rho*a**3)/np.sum(rho),1.0/3.0)
    x0 = (np.sqrt(1+2*kappa*amed)-1)/(2*amed)
    sol = optimize.root(fun, x0, jac=None, method='hybr')
    return sol.x[0]

" The DFT model for electrolyte solutions using the generalized grand potential"

class Electrolyte():
    def __init__(self,N,delta,species=2,a=np.array([1.0,1.0]),Z=np.array([-1,1]),rhob=np.array([0.1,0.1])):
        self.N = N
        self.delta = delta
        self.L = delta*N
        self.a = a
        self.species = species
        self.Z = Z
        self.rhob = rhob
        self.x = np.linspace(0,self.L,self.N)

        self.lB = 0.714 # in nm (for water)
        
        self.n = np.zeros((self.species,self.N),dtype=np.float32)
        self.phi = np.zeros((self.species,self.species,self.N),dtype=np.float32)
        self.phiint = np.zeros((self.species,self.species),dtype=np.float32)
        self.w = np.zeros((self.species,self.N),dtype=np.float32)
        self.Gamma = np.zeros(self.N,dtype=np.float32)
        self.Eta = np.zeros(self.N,dtype=np.float32)

        self.Gammabulk = Gammafuncbulk(self.rhob,self.a,self.Z,self.lB)
        self.Hbulk = np.sum(self.rhob*self.a**3/(1+self.Gammabulk*self.a))+(2.0/np.pi)*(1-(np.pi/6)*np.sum(self.rhob*self.a**3))
        self.Etabulk = np.sum(self.rhob*self.Z*self.a/(1+self.Gammabulk*self.a))/self.Hbulk
        print('inverse Gammabulk = ',1.0/self.Gammabulk,' nm')

        self.b = self.a + 1.0/self.Gammabulk

        for i in range(self.species):
            nsig = int(0.5*self.b[i]/self.delta)
            self.w[i,self.N//2-nsig:self.N//2+nsig] = 1.0/(self.b[i])

            for j in range(self.species):
                bij = 0.5*(self.b[i]+self.b[j])
                aij = 0.5*(self.a[i]+self.a[j])
                nsig = int(aij/self.delta)
                x = np.linspace(-aij,aij,2*nsig)
                self.phi[i,j,self.N//2-nsig:self.N//2+nsig] = -(self.Z[i]*self.Z[j]*self.lB/(bij**2))*phicorr1D(x,bij,aij)
                self.phiint[i,j] = -(self.Z[i]*self.Z[j]*self.lB/(bij**2))*(np.pi*aij**4)*(1-(8/3.0)*bij/aij+2*(bij/aij)**2)

    def auxiliary_quantities(self,rho):
        for i in range(self.species):
            self.n[i,:] = convolve1d(rho[i], weights=self.w[i], mode='nearest')*self.delta

        self.Gamma = Gammafunc(self.n,self.a,self.Z,self.lB) #verified
        self.Eta = Etafunc(self.n,self.a,self.Z,self.Gamma) # verified

    def Flong(self,rho,psi):
        f = convolve1d(psi, weights=[-1,1], mode='nearest')/self.delta
        return -(1/(8*np.pi*self.lB))*np.sum(f**2)*self.delta

    def Fint(self,rho,psi):
        f2 = np.sum(rho[:,:]*self.Z[:,np.newaxis],axis=0)
        return np.sum(f2*psi)*self.delta 

    def Fcorr(self,rho):
        self.auxiliary_quantities(rho)
        Phi = self.Gamma**3/(3*np.pi)
        for i in range(self.species):
            Phi += -self.lB*self.n[i,:]*self.Z[i]*(self.Z[i]*self.Gamma+self.Eta*self.a[i])/(1+self.Gamma*self.a[i])
            for j in range(self.species):
                Phi += 0.5*rho[i,:]*convolve1d(rho[j,:], weights=self.phi[i,j], mode='nearest')*self.delta
        return np.sum(Phi)*self.delta

    def free_energy(self,rho,psi):
        return (self.Flong(rho,psi)+self.Fint(rho,psi)+self.Fcorr(rho))

    def c1MSA(self,rho):
        self.auxiliary_quantities(rho)
        cc = np.zeros((self.species,self.N))
        for i in range(self.species):
            dPhieledn = -self.lB*(self.Z[i]**2*self.Gamma+2*self.a[i]*self.Eta*self.Z[i]-self.Eta**2*self.a[i]**3*(2.0/3.0-self.Gamma*self.a[i]/3.0))/(1+self.Gamma*self.a[i])
            cc[i,:] = -convolve1d(dPhieledn, weights=self.w[i], mode='nearest')*self.delta
        return cc

    def c1nonMSA(self,rho):
        cc = np.zeros((self.species,self.N))
        for i in range(self.species):
            for j in range(self.species):
                cc[i,:] += -convolve1d(rho[j,:], weights=self.phi[i,j], mode='nearest')*self.delta
        return cc

    def c1long(self,psi):
        cc = np.empty((self.species,self.N))
        for i in range(self.species):
            cc[i,:] = -self.Z[i]*psi[:] 
        return cc

    def c1(self,rho,psi):
        return self.c1MSA(rho)+self.c1nonMSA(rho)+self.c1long(psi)

    def dOmegadpsi(self,rho,psi,sigma):
        lappsi = (1/(4*np.pi*self.lB))*convolve1d(psi, weights=[1,-2,1], mode='nearest')/self.delta**2
        f = np.zeros_like(psi)
        f[0] = sigma[0]/self.delta
        f[-1] = sigma[1]/self.delta
        f += np.sum(rho[:,:]*self.Z[:,np.newaxis],axis=0) 
        return lappsi + f

    def muMSA(self,rhob):
        muu = np.zeros_like(rhob)
        for i in range(self.species):
            muu[i] = -self.lB*(self.Z[i]**2*self.Gammabulk+2*self.a[i]*self.Etabulk*self.Z[i]-self.Etabulk**2*self.a[i]**3*(2.0/3.0-self.Gammabulk*self.a[i]/3.0))/(1+self.Gammabulk*self.a[i])
        return muu

    def munonMSA(self,rhob):
        muu = np.zeros_like(rhob)
        for i in range(self.species):
            muu[i] = np.sum(rhob[:]*self.phiint[i,:])
        return muu

    def mu(self,rhob):
        return self.muMSA(rhob)+self.munonMSA(rhob)


if __name__ == "__main__":
    test0 = False # the MSA screening parameter 
    test1 = True
    test2 = False

    import matplotlib.pyplot as plt
    from fire import optimize_fire2
    from fmt import FMTplanar
    from pb import PBplanar

    if test0: 
        rhob = np.linspace(1e-3,5.0,1000)*0.6
        Z = np.array([-2,2])
        a = np.array([0.452,0.452])
        lB = 0.714

        gamma0 = np.zeros_like(rhob)
        gamma1 = np.zeros_like(rhob)
        for i in range(rhob.size):
            rho = np.array([rhob[i],2*rhob[i]])
            kappa = np.sqrt(4*np.pi*lB*np.sum(Z**2*rho))
            # amed = np.power(np.sum(rho*a**3)/np.sum(rho),1.0/3.0)
            amed = np.sqrt(np.sum(rho*a**2)/np.sum(rho))
            gamma0[i] = (np.sqrt(1+2*kappa*amed)-1)/(2*amed)
            gamma1[i] = Gammafunc(rho,a,Z,lB)


        plt.plot(rhob,gamma1,'k')
        plt.plot(rhob,gamma0,'--',color='grey')
        plt.show()
        
    if test1: 
        sigma = np.array([0.3,0.15])
        delta = 0.025*sigma[1]
        L = 12.5*sigma[0]
        N = int(L/delta)
        Z = np.array([-1,3])

        c = 0.1 #mol/L (equivalent to ionic strength for 1:1)
        rhob = np.array([-(Z[1]/Z[0])*c,c])*6.022e23/1.0e24 # particles/nm^3

        x = np.linspace(0,L,N)

        # Gamma = 0.7/sigma[0]**2
        Gamma = -3.12

        n = np.ones((2,N),dtype=np.float32)
        nsig = np.array([int(0.5*sigma[0]/delta),int(0.5*sigma[1]/delta)])

        param = np.array([rhob[0],rhob[1],Gamma])

        # Here we will solve the PB equation as a input to DFT
        pb = PBplanar(N,delta,species=2,sigma=sigma,Z=Z)
        kD = np.sqrt(4*np.pi*pb.lB*np.sum(Z**2*rhob))

        def Fpsi(psi,param):
            n[0,:nsig[0]] = 1.0e-16
            n[1,:nsig[1]] = 1.0e-16
            n[0,nsig[0]:] = param[0]*np.exp(-Z[0]*psi[nsig[0]:])
            n[1,nsig[1]:] = param[1]*np.exp(-Z[1]*psi[nsig[1]:])
            Gamma = param[2]
            Fele = pb.free_energy(n,psi)
            return (Fele+Gamma*psi[0])/L

        def dFpsidpsi(psi,param):
            n[0,:nsig[0]] = 1.0e-16
            n[1,:nsig[1]] = 1.0e-16
            n[0,nsig[0]:] = param[0]*np.exp(-Z[0]*psi[nsig[0]:])
            n[1,nsig[1]:] = param[1]*np.exp(-Z[1]*psi[nsig[1]:])
            Gamma = param[2]
            return -pb.dOmegadpsi(n,psi,Gamma)*delta/L

        psi0 = 0.1*Gamma*4*np.pi*pb.lB # attenuation of the surface charge
        psi = np.zeros(N,dtype=np.float32)
        psi[:nsig[0]] = psi0*(1/kD+0.5*sigma[0]-x[:nsig[0]])
        psi[nsig[0]:] = psi0*np.exp(-kD*(x[nsig[0]:]-0.5*sigma[0]))/kD
    
        [varsol,Omegasol,Niter] = optimize_fire2(psi,Fpsi,dFpsidpsi,param,1.0e-8,0.02,False)

        psi[:] = varsol

        n[0,:nsig[0]] = 1.0e-16
        n[1,:nsig[1]] = 1.0e-16
        n[0,nsig[0]:] = param[0]*np.exp(-Z[0]*psi[nsig[0]:])
        n[1,nsig[1]:] = param[1]*np.exp(-Z[1]*psi[nsig[1]:])

        # Now we will solve the DFT with electrostatic correlations
        nn = n.copy()

        fmt = FMTplanar(N,delta,species=2,sigma=sigma)
        ele = Electrolyte(N,delta,species=2,a=sigma,Z=Z,rhob=rhob)

        # solving the electrostatic potential equation
        def Fpsi2(psi,nn):
            Fele = ele.Flong(nn,psi)+ele.Fint(nn,psi)
            return (Fele+Gamma*psi[0])/L

        def dFpsidpsi2(psi,nn):
            return -ele.dOmegadpsi(nn,psi,[Gamma,0.0])*delta/L

        mu = np.log(rhob) + fmt.mu(rhob) + ele.mu(rhob)
        # mu = np.log(rhob) + fmt.mu(rhob)

        # Now we will solve the DFT equations
        def Omega(var,psi):
            nn[0,:] = np.exp(var[0])
            nn[1,:] = np.exp(var[1])
            Fid = np.sum(nn*(var-1.0))*delta
            Fhs = np.sum(fmt.Phi(nn))*delta
            Fele = ele.Fint(n,psi) + ele.Fcorr(n)
            # Fele = ele.Fint(n,psi)
            # Fele = ele.free_energy(n,psi)
            return (Fid+Fhs+Fele-np.sum(mu[:,np.newaxis]*nn*delta)+Gamma*psi[0])/L

        def dOmegadnR(var,psi):
            nn[0,:] = np.exp(var[0])
            nn[1,:] = np.exp(var[1])

            [varsol2,Omegasol2,Niter] = optimize_fire2(psi,Fpsi2,dFpsidpsi2,nn,1.0e-6,0.02,False)
            psi[:] = varsol2-varsol2[-1]

            c1hs = fmt.c1(nn)
            c1ele = ele.c1(nn,psi)
            # c1ele = ele.c1long(psi)
            aux = nn*(var -c1hs -c1ele - mu[:,np.newaxis])*delta/L
            aux[0,-nsig[0]:] = 0.0
            aux[1,-nsig[1]:] = 0.0
            return aux

        var = np.log(n)
        [varsol,Omegasol1,Niter] = optimize_fire2(var,Omega,dOmegadnR,psi,1.0e-5,0.02,True)
        n[0,:] = np.exp(varsol[0])
        n[1,:] = np.exp(varsol[1])

        muMSA = ele.muMSA(rhob)
        munonMSA = ele.munonMSA(rhob)
        print('muMSA =',muMSA)
        print('munonMSA =',munonMSA)

        c1MSA = ele.c1MSA(n)+muMSA[:,np.newaxis]
        # print(c1MSA)
        c1nonMSA = ele.c1nonMSA(n)+munonMSA[:,np.newaxis]
        # print(c1nonMSA)

        # np.save('profiles-DFTcorr-electrolyte22-c0.5-sigma-0.1704.npy',[x,n[0],n[1],psi])
        np.save('profiles-DFTcorr-Voukadinova2018-electrolyte-Fig3-Z+=3-rho+=0.1M.npy',[x,n[0],n[1],psi,c1MSA[0],c1MSA[1],c1nonMSA[0],c1nonMSA[1]])

    ##################################################################################
    if test2: 
        sigma = np.array([0.3,0.3])
        delta = 0.025*sigma[1]
        Z = np.array([-1,3])
        c = 0.01 #mol/L (equivalent to ionic strength for 1:1)
        rhob = np.array([-(Z[1]/Z[0])*c,c])*6.022e23/1.0e24 # particles/nm^3
        Gamma = -3.12

        [x,nani,ncat,psi] = np.load('profiles-DFTcorr-Voukadinova2018-electrolyte11-Fig5-Z+=3-rho+=0.01M.npy')
        N = x.size

        n = np.ones((2,N),dtype=np.float32)
        n[0] = nani
        n[1] = ncat

        # fmt = FMTplanar(N,delta,species=2,sigma=sigma)
        ele = Electrolyte(N,delta,species=2,a=sigma,Z=Z,rhob=rhob)

        muMSA = ele.muMSA(rhob)
        munonMSA = ele.munonMSA(rhob)
        print(muMSA)
        print(munonMSA)

        c1MSA = ele.c1MSA(n)+muMSA[:,np.newaxis]
        # print(c1MSA)
        c1nonMSA = ele.c1nonMSA(n)+munonMSA[:,np.newaxis]
        # print(c1nonMSA)

        np.save('profiles-DFTcorr-Voukadinova2018-electrolyte11-correlation-Fig5-Z+=3-rho+=0.01M.npy',[x,c1MSA[0],c1MSA[1],c1nonMSA[0],c1nonMSA[1]])  