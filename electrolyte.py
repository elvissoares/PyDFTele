#!/usr/bin/env python3

# This script is the python implementation of the Density Functional Theory
# for Electrolyte Solution in the presence of an external electrostatic potential
#
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2020-06-02
# Updated: 2021-07-19
# Version: 0.1
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

def PhiMSA(rho,a,Z,lB,eta,Gamma):
    aux = Gamma**3/(3*np.pi)
    aux += -lB*np.sum(rho*Z*(Z*Gamma+a*eta)/(1+Gamma*a))
    return aux

def dPhiMSAdrho(a,Z,lB,eta,Gamma):
    return -lB*(Z**2*Gamma+2*a*eta*Z-eta**2*a**3*(2.0/3.0-Gamma*a/3.0))/(1+Gamma*a)

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

        self.b = self.a + 1.0/self.Gammabulk

        for i in range(self.species):
            nsig = int(0.5*self.b[i]/self.delta)
            self.w[i,self.N//2-nsig:self.N//2+nsig] = 1.0/(self.b[i])

            for j in range(self.species):
                bij = 0.5*(self.b[i]+self.b[j])
                aij = 0.5*(self.a[i]+self.a[j])
                nsig = int(aij/self.delta)
                x = np.linspace(-aij,aij,2*nsig,)
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
            # dPhieledn = -self.lB*(self.Z[i]**2*self.Gamma+2*self.a[i]*self.Eta*self.Z[i])/(1+self.Gamma*self.a[i])
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
        # lappsi[-1] = 0.0
        return lappsi + f

    def muMSA(self,rhob):
        muu = np.zeros_like(rhob)
        for i in range(self.species):
            muu[i] = -self.lB*(self.Z[i]**2*self.Gammabulk+2*self.a[i]*self.Etabulk*self.Z[i]-self.Etabulk**2*self.a[i]**3*(2.0/3.0-self.Gammabulk*self.a[i]/3.0))/(1+self.Gammabulk*self.a[i])
        # muu = -self.lB*(self.Z**2*self.Gammabulk+2*self.a*self.Etabulk*self.Z)/(1+self.Gammabulk*self.a)
        # for i in range(self.species):
        return muu

    def munonMSA(self,rhob):
        muu = np.zeros_like(rhob)
        for i in range(self.species):
            muu[i] = np.sum(rhob[:]*self.phiint[i,:])
        return muu

    def mu(self,rhob):
        return self.muMSA(rhob)+self.munonMSA(rhob)

# sucessive over-relaxation
def sor(rho,psi,Z,sigma,delta,lB,L):
    omega = 0.9
    error = 1.0
    f = rho[0,:]*Z[0]+rho[1,:]*Z[1]
    f[0] = sigma[0]/delta
    f[-1] = sigma[1]/delta
    while error > 5.e-3:
        psidiff = convolve1d(psi, weights=[1,0,1], mode='nearest')
        psinew = (1-omega)*psi + 0.5*omega*(psidiff +(4*np.pi*lB)*f*delta**2)
        error = np.max(np.abs((psidiff-2*psi +(4*np.pi*lB)*f*delta**2)))
        psi[:] = psinew 
        # psi[-1] = 0.0
        print(error)
        # x = np.linspace(0,L,psi.size)
        # plt.plot(x,psidiff)
        # plt.plot(x,(4*np.pi*lB)*f*delta**2)
        # plt.show()
    return psi


if __name__ == "__main__":
    test0 = False # the MSA screening parameter 
    test1 = False
    test2 = False
    test3 = False
    test4 = True
    test5 = False
    test6 = False

    import matplotlib.pyplot as plt
    from fire import optimize_fire2
    from fmt import FMTplanar

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
        sigma = np.array([0.425,0.425])
        delta = 0.025*min(sigma)
        N = 400
        L = N*delta
        beta = 1.0/40.0
        Z = np.array([-1,1])

        c = 1.0 #mol/L (equivalent to ionic strength for 1:1)
        rhob = np.array([c,c])*6.022e23/1.0e24 # particles/nm^3

        fmt = FMTplanar(N,delta,species=2,sigma=sigma)
        ele = Electrolyte(N,delta,species=2,a=sigma,Z=Z)
        x = np.linspace(0,L,N)

        Gamma = 0.7/sigma[0]**2

        n = np.ones((2,N),dtype=np.float32)
        nsig = np.array([int(0.5*sigma[0]/delta),int(0.5*sigma[1]/delta)])

        kD = np.sqrt(4*np.pi*ele.lB*np.sum(Z**2*rhob))

        # solving the regular PB equation
        def Fpsi(psi,param):
            n[0,:nsig[0]] = 1.0e-16
            n[1,:nsig[1]] = 1.0e-16
            n[0,nsig[0]:] = param[0]*np.exp(-Z[0]*psi[nsig[0]:])
            n[1,nsig[1]:] = param[1]*np.exp(-Z[1]*psi[nsig[1]:])
            n[0,-nsig[0]:] = 1.0e-16
            n[1,-nsig[1]:] = 1.0e-16
            Gamma = param[2]
            Fele = ele.Flong(n,psi)
            return (Fele+Gamma*psi[0])/L

        def dFpsidpsi(psi,param):
            n[0,:nsig[0]] = 1.0e-16
            n[1,:nsig[1]] = 1.0e-16
            n[0,nsig[0]:] = param[0]*np.exp(-Z[0]*psi[nsig[0]:])
            n[1,nsig[1]:] = param[1]*np.exp(-Z[1]*psi[nsig[1]:])
            n[0,-nsig[0]:] = 1.0e-16
            n[1,-nsig[1]:] = 1.0e-16
            Gamma = param[2]
            return -ele.dOmegadpsi(n,psi,[Gamma,Gamma])*delta/L

        param = np.array([rhob[0],rhob[1],Gamma])

        psi = np.zeros_like(x)
        psi0 = Gamma*4*np.pi*ele.lB
        psi = psi0*np.exp(-kD*x)+psi0*np.exp(+kD*(x-L))
    
        [varsol,Omegasol,Niter] = optimize_fire2(psi,Fpsi,dFpsidpsi,param,1.0e-8,0.02,False)

        psi[:] = varsol
        n[0,:nsig[0]] = 1.0e-16
        n[1,:nsig[1]] = 1.0e-16
        n[0,nsig[0]:] = rhob[0]*np.exp(-Z[0]*psi[nsig[0]:])
        n[1,nsig[1]:] = rhob[1]*np.exp(-Z[1]*psi[nsig[1]:])
        n[0,-nsig[0]:] = 1.0e-16
        n[1,-nsig[1]:] = 1.0e-16

        # n[0,:nsig[0]] = 1.0e-16
        # n[1,:nsig[1]] = 1.0e-16
        # n[0,nsig[0]:] = rhob[0]
        # n[1,nsig[1]:] = rhob[1]

        # Now we will solve the DFT equations
        var = np.array([np.log(n[0]),np.log(n[1]),psi])
        aux = np.zeros_like(var)

        def Omega(var,param):
            n[0,:] = np.exp(var[0])
            n[1,:] = np.exp(var[1])
            psi[:] = var[2]
            mu[:] = param[0:2]
            Gamma = param[2]
            Fid = np.sum(n*(var[0:2,:]-1.0))*delta
            Fhs = np.sum(fmt.Phi(n))*delta
            Fele = ele.free_energy(n,psi)
            return (Fid+Fhs+Fele-np.sum(mu[:,np.newaxis]*n)*delta + Gamma*psi[0])/L
        
        def dOmegadnR(var,param):
            n[0,:] = np.exp(var[0])
            n[1,:] = np.exp(var[1])
            psi[:] = var[2]
            mu[:] = param[0:2]
            Gamma = param[2]
            c1hs = fmt.c1(n)
            c1ele = ele.c1(n,psi)
            aux[0,:] = n[0]*(var[0] -c1hs[0] -c1ele[0] - mu[0])*delta/L
            aux[1,:] = n[1]*(var[1] -c1hs[1] -c1ele[1] - mu[1])*delta/L
            aux[2,:] = -ele.dOmegadpsi(n,psi,[Gamma,0])*delta/L
            return aux

        mu = np.log(rhob) + fmt.mu(rhob) + ele.mu(rhob)

        print('mu =',mu)
        print('rhob=',rhob)

        param = np.array([mu[0],mu[1],Gamma])
    
        [varsol,Omegasol,Niter] = optimize_fire2(var,Omega,dOmegadnR,param,1.0e-3,0.05,True)

        n[0,:] = np.exp(varsol[0])
        n[1,:] = np.exp(varsol[1])
        psi[:] = varsol[2]
        nmean = np.sum(n,axis=1)*delta/L
        print('mu =',mu)
        print('rhob=',rhob,'\n nmean = ',nmean,'\n Omega/L =',Omegasol)

        np.save('profiles-DFTcorr-electrolyte11-c1.0-sigma0.7.npy',[x,n[0],n[1],psi])

    if test2: 
        sigma = np.array([0.3,0.3])
        delta = 0.025*sigma[0]
        N = 200
        L = N*delta
        Z = np.array([-1,1])

        c = 1.0 #mol/L (equivalent to ionic strength for 1:1)
        rhob = np.array([-(Z[1]/Z[0])*c,c])*6.022e23/1.0e24 # particles/nm^3

        fmt = FMTplanar(N,delta,species=2,sigma=sigma)
        ele = Electrolyte(N,delta,species=2,a=sigma,Z=Z,rhob=rhob)
        x = np.linspace(0,L,N)

        # Gamma = 0.7/sigma[0]**2
        Gamma = -3.12
        kD = np.sqrt(4*np.pi*ele.lB*np.sum(Z**2*rhob))

        print('sigma=',Gamma)

        n = np.ones((2,N),dtype=np.float32)
        nsig = np.array([int(0.5*sigma[0]/delta),int(0.5*sigma[1]/delta)])

        param = np.array([rhob[0],rhob[1],Gamma])

        psi0 = Gamma*4*np.pi*ele.lB
        psi = np.zeros(N,dtype=np.float32)

        n[0,:nsig[0]] = 1.0e-16
        n[1,:nsig[1]] = 1.0e-16
        n[0,nsig[0]:] = rhob[0]
        n[1,nsig[1]:] = rhob[1]

        # solving the regular PB equation
        def Fpsi(psi,param):
            n[0,:nsig[0]] = 1.0e-16
            n[1,:nsig[1]] = 1.0e-16
            n[0,nsig[0]:] = param[0]*np.exp(-Z[0]*psi[nsig[0]:])
            n[1,nsig[1]:] = param[1]*np.exp(-Z[1]*psi[nsig[1]:])
            Gamma = param[2]
            Fele = ele.Flong(n,psi)
            return (Fele+Gamma*psi[0])/L

        def dFpsidpsi(psi,param):
            n[0,:nsig[0]] = 1.0e-16
            n[1,:nsig[1]] = 1.0e-16
            n[0,nsig[0]:] = param[0]*np.exp(-Z[0]*psi[nsig[0]:])
            n[1,nsig[1]:] = param[1]*np.exp(-Z[1]*psi[nsig[1]:])
            Gamma = param[2]
            return -ele.dOmegadpsi(n,psi,[Gamma,0.0])*delta/L

        param = np.array([rhob[0],rhob[1],Gamma])

        psi0 = Gamma*4*np.pi*ele.lB
        psi[:nsig[0]] = psi0*(1/kD+0.5*sigma[0]-x[:nsig[0]])
        psi[nsig[0]:] = psi0*np.exp(-kD*(x[nsig[0]:]-0.5*sigma[0]))/kD
    
        # [varsol,Omegasol,Niter] = optimize_fire2(psi,Fpsi,dFpsidpsi,param,1.0e-8,0.02,False)
        # psi[:] = varsol
        # n[0,:nsig[0]] = 1.0e-16
        # n[1,:nsig[1]] = 1.0e-16
        # n[0,nsig[0]:] = rhob[0]*np.exp(-Z[0]*psi[nsig[0]:])
        # n[1,nsig[1]:] = rhob[1]*np.exp(-Z[1]*psi[nsig[1]:])

        # plt.plot(x,psi)
        # plt.show()
        # plt.plot(x,n[0]/rhob[0])
        # plt.plot(x,n[1]/rhob[1])
        # plt.show()

        nn = n.copy()
        var2 = np.empty_like(n)

        mu = np.log(rhob) + fmt.mu(rhob) + ele.mu(rhob)

        # Now we will solve the DFT equations
        def Omega(var,psii):
            nn[0,:] = np.exp(var[0])
            nn[1,:] = np.exp(var[1])
            Fid = np.sum(nn*(var-1.0))*delta
            Fhs = np.sum(fmt.Phi(nn))*delta
            Fele = ele.Fint(nn,psii)+ele.Fcorr(nn)
            return (Fid+Fhs+Fele-np.sum(mu[:,np.newaxis]*nn)*delta+Gamma*psii[0])/L
        
        def dOmegadnR(var,psii):
            nn[0,:] = np.exp(var[0])
            nn[1,:] = np.exp(var[1])
            c1hs = fmt.c1(nn)
            c1ele = ele.c1(nn,psii)
            return nn*(var -c1hs -c1ele - mu[:,np.newaxis])*delta/L

        # solving the regular PB equation
        def Fpsi2(psii,nn):
            Fele = ele.Fint(nn,psii)+ele.Flong(nn,psii)
            return (Fele+Gamma*psii[0])/L

        def dFpsidpsi2(psii,nn):
            return -ele.dOmegadpsi(nn,psii,[Gamma,0])*delta/L

        
        Fid = np.sum(n*(np.log(n)-1.0))*delta
        Fhs = np.sum(fmt.Phi(n))*delta
        Fele = ele.free_energy(n,psi)
        Omegasol=(Fid+Fhs+Fele-np.sum(mu[:,np.newaxis]*n)*delta+Gamma*psi[0])/L

        # n[0,nsig[0]:] = rhob[0]
        # n[1,nsig[1]:] = rhob[1]

        error = 1.0
        
        while error > 1e-4:

            var = np.log(n)
            [varsol,Omegasol1,Niter] = optimize_fire2(var,Omega,dOmegadnR,psi,1.0e-6,0.002,True)

            n[0,:] = np.exp(varsol[0])
            n[1,:] = np.exp(varsol[1])

            [varsol2,Omegasol2,Niter] = optimize_fire2(psi,Fpsi2,dFpsidpsi2,n,1.0e-6,0.02,False)
            # psi = sor(ele.n,psi,ele.Z,[Gamma,0],ele.delta,ele.lB,L)

            psi[:] = varsol2-varsol2[-1]

            Omegalast = Omegasol

            Fid = np.sum(n*(np.log(n)-1.0))*delta
            Fhs = np.sum(fmt.Phi(n))*delta
            Fele = ele.free_energy(n,psi)
            Omegasol=(Fid+Fhs+Fele-np.sum(mu[:,np.newaxis]*n*delta)+Gamma*psi[0])/L

            # plt.plot(x,n[0]/rhob[0])
            # plt.plot(x,n[1]/rhob[1])
            # plt.show()

            # plt.plot(x,psi)
            # plt.show()

            error = abs(Omegasol-Omegalast)
            print(error)

        # np.save('profiles-DFTcorr-electrolyte22-c0.5-sigma-0.1704.npy',[x,n[0],n[1],psi])
        np.save('profiles-DFTcorr-Voukadinova2018-electrolyte11-Fig5-Z+=1-rho+=1.0M.npy',[x,n[0],n[1],psi])

    if test3: 
        sigma = np.array([0.425,0.425])
        delta = 0.025*min(sigma)
        N = 600
        L = N*delta
        Z = np.array([-1,1])

        c = 1.0 #mol/L (equivalent to ionic strength for 1:1)
        rhob = np.array([c,c])*6.022e23/1.0e24 # particles/nm^3

        fmt = FMTplanar(N,delta,species=2,sigma=sigma)
        ele = Electrolyte(N,delta,species=2,a=sigma,Z=Z,rhob=rhob)
        x = np.linspace(0,L,N)

        Gamma = 0.7/sigma[0]**2
        kD = np.sqrt(4*np.pi*ele.lB*np.sum(Z**2*rhob))

        n = np.ones((2,N),dtype=np.float32)
        nsig = np.array([int(0.5*sigma[0]/delta),int(0.5*sigma[1]/delta)])

        param = np.array([rhob[0],rhob[1],Gamma])

        psi0 = Gamma*4*np.pi*ele.lB
        psi = np.zeros(N,dtype=np.float32)
        
        n[0,:nsig[0]] = 1.0e-16
        n[1,:nsig[1]] = 1.0e-16
        n[0,nsig[0]:] = rhob[0]
        n[1,nsig[1]:] = rhob[1]

        # solving the regular PB equation
        def Fpsi(psi,param):
            n[0,:nsig[0]] = 1.0e-16
            n[1,:nsig[1]] = 1.0e-16
            n[0,nsig[0]:] = param[0]*np.exp(-Z[0]*psi[nsig[0]:])
            n[1,nsig[1]:] = param[1]*np.exp(-Z[1]*psi[nsig[1]:])
            Gamma = param[2]
            Fele = ele.Flong(n,psi)+ele.Fint(n,psi)
            return (Fele+Gamma*psi[0])/L

        def dFpsidpsi(psi,param):
            n[0,:nsig[0]] = 1.0e-16
            n[1,:nsig[1]] = 1.0e-16
            n[0,nsig[0]:] = param[0]*np.exp(-Z[0]*psi[nsig[0]:])
            n[1,nsig[1]:] = param[1]*np.exp(-Z[1]*psi[nsig[1]:])
            Gamma = param[2]
            return -ele.dOmegadpsi(n,psi,[Gamma,0.0])*delta/L

        param = np.array([rhob[0],rhob[1],Gamma])

        psi0 = Gamma*4*np.pi*ele.lB
        psi[:nsig[0]] = psi0*(1/kD+0.5*sigma[0]-x[:nsig[0]])
        psi[nsig[0]:] = psi0*np.exp(-kD*(x[nsig[0]:]-0.5*sigma[0]))/kD
        # psi[:] = psi0*np.exp(-kD*x)
    
        [varsol,Omegasol,Niter] = optimize_fire2(psi,Fpsi,dFpsidpsi,param,1.0e-8,0.02,False)

        psi[:] = varsol
        n[0,:nsig[0]] = 1.0e-16
        n[1,:nsig[1]] = 1.0e-16
        n[0,nsig[0]:] = rhob[0]*np.exp(-Z[0]*psi[nsig[0]:])
        n[1,nsig[1]:] = rhob[1]*np.exp(-Z[1]*psi[nsig[1]:])

        # plt.plot(x,psi)
        # plt.show()
        # plt.plot(x,n[0]/rhob[0])
        # plt.plot(x,n[1]/rhob[1])
        # plt.show()

        nn = n.copy()
        var2 = np.empty_like(n)

        mu = np.log(rhob) + fmt.mu(rhob) + ele.mu(rhob)

        # Now we will solve the DFT equations
        def Omega(var,psi):
            nn[0,:] = np.exp(var[0])
            nn[1,:] = np.exp(var[1])
            Fid = np.sum(nn*(var-1.0))*delta
            Fhs = np.sum(fmt.Phi(n))*delta
            Fele = ele.Fint(nn,psi)+ele.Fcorr(nn)
            return (Fid+Fhs+Fele-np.sum(mu[:,np.newaxis]*nn)*delta+Gamma*psi[0])/L
        
        def dOmegadnR(var,psi):
            nn[0,:] = np.exp(var[0])
            nn[1,:] = np.exp(var[1])
            c1hs = fmt.c1(nn)
            c1ele = ele.c1(nn,psi)
            return nn*(var -c1hs -c1ele - mu[:,np.newaxis])*delta/L

        # solving the regular PB equation
        def Fpsi2(psii,var):
            nn[0,:nsig[0]] = 1.0e-16
            nn[1,:nsig[1]] = 1.0e-16
            nn[0,nsig[0]:] = var[0,nsig[0]:]*np.exp(-Z[0]*psii[nsig[0]:])
            nn[1,nsig[1]:] = var[1,nsig[1]:]*np.exp(-Z[1]*psii[nsig[1]:])
            Fele = ele.Flong(nn,psii)+ele.Fint(nn,psii)
            return (Fele+Gamma*psii[0])/L

        def dFpsidpsi2(psii,var):
            nn[0,:nsig[0]] = 1.0e-16
            nn[1,:nsig[1]] = 1.0e-16
            nn[0,nsig[0]:] = var[0,nsig[0]:]*np.exp(-Z[0]*psii[nsig[0]:])
            nn[1,nsig[1]:] = var[1,nsig[1]:]*np.exp(-Z[1]*psii[nsig[1]:])
            return -ele.dOmegadpsi(nn,psii,[Gamma,0])*delta/L

        Fid = np.sum(n*(np.log(n)-1.0))*delta
        Fhs = np.sum(fmt.Phi(n))*delta
        Fele = ele.free_energy(n,psi)
        Omegasol=(Fid+Fhs+Fele-np.sum(mu[:,np.newaxis]*n)*delta+Gamma*psi[0])/L

        # Omegasol = 0.0
        error = 1.0
        
        while error > 1e-4:

            var = np.log(n)
            [varsol,Omegasol1,Niter] = optimize_fire2(var,Omega,dOmegadnR,psi,1.0e-4,0.02,True)

            n[:] = np.exp(varsol)

            var2[0,nsig[0]:] = n[0,nsig[0]:]/np.exp(-Z[0]*psi[nsig[0]:])
            var2[1,nsig[1]:] = n[1,nsig[1]:]/np.exp(-Z[1]*psi[nsig[1]:])

            [varsol2,Omegasol2,Niter] = optimize_fire2(psi,Fpsi2,dFpsidpsi2,var2,1.0e-8,0.02,False)

            psi[:] = varsol2

            n[0,:nsig[0]] = 1.0e-16
            n[1,:nsig[1]] = 1.0e-16
            n[0,nsig[0]:] = var2[0,nsig[0]:]*np.exp(-Z[0]*psi[nsig[0]:])
            n[1,nsig[1]:] = var2[1,nsig[1]:]*np.exp(-Z[1]*psi[nsig[1]:])

        
            Omegalast = Omegasol

            Fid = np.sum(n*(np.log(n)-1.0))*delta
            Fhs = np.sum(fmt.Phi(n))*delta
            Fele = ele.free_energy(n,psi)
            Omegasol=(Fid+Fhs+Fele-np.sum(mu[:,np.newaxis]*n*delta)+Gamma*psi[0])/L

            plt.plot(x,n[0]/rhob[0])
            plt.plot(x,n[1]/rhob[1])
            plt.show()

            plt.plot(x,psi)
            plt.show()

            error = abs(Omegasol-Omegalast)
            print(error)

        # np.save('profiles-DFTcorr-electrolyte22-c0.5-sigma-0.1704.npy',[x,n[0],n[1],psi])
        np.save('profiles-DFTcorr-electrolyte11-c1.0-sigma0.7.npy',[x,n[0],n[1],psi])    
        
    if test4: 
        sigma = np.array([0.3,0.3])
        delta = 0.025*sigma[1]
        L = 5*sigma[0]
        N = int(L/delta)
        Z = np.array([-1,3])

        c = 1.0 #mol/L (equivalent to ionic strength for 1:1)
        rhob = np.array([-(Z[1]/Z[0])*c,c])*6.022e23/1.0e24 # particles/nm^3

        fmt = FMTplanar(N,delta,species=2,sigma=sigma)
        ele = Electrolyte(N,delta,species=2,a=sigma,Z=Z,rhob=rhob)
        x = np.linspace(0,L,N)

        # Gamma = 0.7/sigma[0]**2
        Gamma = -3.12
        kD = np.sqrt(4*np.pi*ele.lB*np.sum(Z**2*rhob))

        n = np.ones((2,N),dtype=np.float32)
        nsig = np.array([int(0.5*sigma[0]/delta),int(0.5*sigma[1]/delta)])

        param = np.array([rhob[0],rhob[1],Gamma])

        psi0 = Gamma*4*np.pi*ele.lB
        psi = np.zeros(N,dtype=np.float32)
        psi[:nsig[0]] = psi0*(1/kD+0.5*sigma[0]-x[:nsig[0]])
        psi[nsig[0]:] = psi0*np.exp(-kD*(x[nsig[0]:]-0.5*sigma[0]))/kD
        # psi[:] = psi0*np.exp(-kD*x)
    
        n[0,:nsig[0]] = 1.0e-16
        n[1,:nsig[1]] = 1.0e-16
        n[0,nsig[0]:] = rhob[0]
        n[1,nsig[1]:] = rhob[1]

        # solving the regular PB equation
        def Fpsi(psi,param):
            n[0,:nsig[0]] = 1.0e-16
            n[1,:nsig[1]] = 1.0e-16
            n[0,nsig[0]:] = param[0]*np.exp(-Z[0]*psi[nsig[0]:])
            n[1,nsig[1]:] = param[1]*np.exp(-Z[1]*psi[nsig[1]:])
            Gamma = param[2]
            Fele = ele.Flong(n,psi) + ele.Fint(n,psi)
            return (Fele+Gamma*psi[0])/L

        def dFpsidpsi(psi,param):
            n[0,:nsig[0]] = 1.0e-16
            n[1,:nsig[1]] = 1.0e-16
            n[0,nsig[0]:] = param[0]*np.exp(-Z[0]*psi[nsig[0]:])
            n[1,nsig[1]:] = param[1]*np.exp(-Z[1]*psi[nsig[1]:])
            Gamma = param[2]
            return -ele.dOmegadpsi(n,psi,[Gamma,0.0])*delta/L

        param = np.array([rhob[0],rhob[1],Gamma])
            
        # [varsol,Omegasol,Niter] = optimize_fire2(psi,Fpsi,dFpsidpsi,param,1.0e-8,0.02,False)
        # psi[:] = varsol

        nn = n.copy()
        var2 = n.copy()

        # solving the regular PB equation
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
            return nn*(var -c1hs -c1ele - mu[:,np.newaxis])*delta/L

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
        np.save('profiles-DFTcorr-Voukadinova2018-electrolyte-Fig5-Z+=3-rho+=1.0M.npy',[x,n[0],n[1],psi,c1MSA[0],c1MSA[1],c1nonMSA[0],c1nonMSA[1]])

    if test5: 
        sigma = np.array([0.3,0.3])
        delta = 0.025*sigma[0]
        N = 400
        L = N*delta
        Z = np.array([-1,3])

        c = 1.0 #mol/L (equivalent to ionic strength for 1:1)
        rhob = np.array([-(Z[1]/Z[0])*c,c])*6.022e23/1.0e24 # particles/nm^3

        fmt = FMTplanar(N,delta,species=2,sigma=sigma)
        ele = Electrolyte(N,delta,species=2,a=sigma,Z=Z,rhob=rhob)
        x = np.linspace(0,L,N)

        # Gamma = 0.396/sigma[0]**2
        Gamma = -3.12
        kD = np.sqrt(4*np.pi*ele.lB*np.sum(Z**2*rhob))

        n = np.ones((2,N),dtype=np.float32)
        nsig = np.array([int(0.5*sigma[0]/delta),int(0.5*sigma[1]/delta)])

        param = np.array([rhob[0],rhob[1],Gamma])

        psi0 = Gamma*4*np.pi*ele.lB
        psi = np.zeros(N,dtype=np.float32)
        psi[:nsig[0]] = psi0*(1/kD+0.5*sigma[0]-x[:nsig[0]])
        psi[nsig[0]:] = psi0*np.exp(-kD*(x[nsig[0]:]-0.5*sigma[0]))/kD
        # psi[:] = psi0*np.exp(-kD*x)
    
        n[0,:nsig[0]] = 1.0e-16
        n[1,:nsig[1]] = 1.0e-16
        n[0,nsig[0]:] = rhob[0]
        n[1,nsig[1]:] = rhob[1]

        # solving the regular PB equation
        def Fpsi(psi,param):
            n[0,:nsig[0]] = 1.0e-16
            n[1,:nsig[1]] = 1.0e-16
            n[0,nsig[0]:] = param[0]*np.exp(-Z[0]*psi[nsig[0]:])
            n[1,nsig[1]:] = param[1]*np.exp(-Z[1]*psi[nsig[1]:])
            Gamma = param[2]
            Fele = ele.Flong(n,psi) + ele.Fint(n,psi)
            return (Fele+Gamma*psi[0])/L

        def dFpsidpsi(psi,param):
            n[0,:nsig[0]] = 1.0e-16
            n[1,:nsig[1]] = 1.0e-16
            n[0,nsig[0]:] = param[0]*np.exp(-Z[0]*psi[nsig[0]:])
            n[1,nsig[1]:] = param[1]*np.exp(-Z[1]*psi[nsig[1]:])
            Gamma = param[2]
            return -ele.dOmegadpsi(n,psi,[Gamma,0.0])*delta/L

        param = np.array([rhob[0],rhob[1],Gamma])
            
        # [varsol,Omegasol,Niter] = optimize_fire2(psi,Fpsi,dFpsidpsi,param,1.0e-8,0.02,False)
        # psi[:] = varsol

        # n[0,nsig[0]:] = rhob[0]*np.exp(-Z[0]*psi[nsig[0]:])
        # n[1,nsig[1]:] = rhob[1]*np.exp(-Z[1]*psi[nsig[1]:])

        # plt.plot(x,psi)
        # plt.show()
        # plt.plot(x/sigma[0],n[0]/rhob[0])
        # plt.plot(x/sigma[0],n[1]/rhob[1])
        # plt.ylim(0,8)
        # plt.xlim(0.5,4.5)
        # plt.show()

        nnew = n.copy()
        nn = n.copy()
        var2 = n.copy()

        mu = np.log(rhob) + fmt.mu(rhob) + ele.mu(rhob)

        # solving the regular PB equation
        def Fpsi2(psi,nn):
            # nn[0,:nsig[0]] = 1.0e-16
            # nn[1,:nsig[1]] = 1.0e-16
            # nn[0,nsig[0]:] = var[0,nsig[0]:]*np.exp(-Z[0]*psi[nsig[0]:])
            # nn[1,nsig[1]:] = var[1,nsig[1]:]*np.exp(-Z[1]*psi[nsig[1]:])
            Fele = ele.Flong(nn,psi)+ele.Fint(nn,psi)
            return (Fele+Gamma*psi[0])/L

        def dFpsidpsi2(psi,nn):
            # nn[0,:nsig[0]] = 1.0e-16
            # nn[1,:nsig[1]] = 1.0e-16
            # nn[0,nsig[0]:] = var[0,nsig[0]:]*np.exp(-Z[0]*psi[nsig[0]:])
            # nn[1,nsig[1]:] = var[1,nsig[1]:]*np.exp(-Z[1]*psi[nsig[1]:])
            return -ele.dOmegadpsi(nn,psi,[Gamma,0.0])*delta/L

        # Now we will solve the DFT equations
        def Omega(var,psi):
            nn[0,:] = np.exp(var[0])
            nn[1,:] = np.exp(var[1])
            Fid = np.sum(nn*(var-1.0))*delta
            Fhs = np.sum(fmt.Phi(nn))*delta
            Fele = ele.free_energy(nn,psi)
            return (Fid+Fhs+Fele-np.sum(mu[:,np.newaxis]*nn*delta)+Gamma*psi[0])/L

        error = 1.0
        alpha = 0.2

        Omegasol = 10000

        muexc = fmt.mu(rhob) + ele.mu(rhob)
        i= 0

        while error > 1e-3:

            

            # if (i> 20):
            #     var2[0,nsig[0]:] = n[0,nsig[0]:]/np.exp(-Z[0]*psi[nsig[0]:])
            #     var2[1,nsig[1]:] = n[1,nsig[1]:]/np.exp(-Z[1]*psi[nsig[1]:])

            #     [varsol2,Omegasol2,Niter] = optimize_fire2(psi,Fpsi2,dFpsidpsi2,var2,1.0e-8,0.02,False)

            #     n[0,nsig[0]:] = var2[0,nsig[0]:]*np.exp(-Z[0]*psi[nsig[0]:])
            #     n[1,nsig[1]:] = var2[1,nsig[1]:]*np.exp(-Z[1]*psi[nsig[1]:])

            #     psi[:] = varsol2-varsol2[-1]

            #     i = 0

            # psi = sor(n,psi,ele.Z,[Gamma,0],ele.delta,ele.lB,L)
            # f = n[0,:]*Z[0]+n[1,:]*Z[1]
            # f[0] = Gamma/delta
            # psidiff = convolve1d(psi, weights=[1,0,1], mode='nearest')
            # psinew = (1-alpha)*psi + 0.5*alpha*(psidiff +(4*np.pi*ele.lB)*f*delta**2)

            # psi = psinew

            # the Picard algorithm
            c1hs = fmt.c1(n)
            c1ele = ele.c1(n,psi)
            # plt.plot(x,c1ele[0])
            # plt.plot(x,c1ele[1])
            # plt.show()
            nnew[0,nsig[0]:] = (1-alpha)*n[0,nsig[0]:] + alpha*rhob[0]*np.exp(c1hs[0,nsig[0]:] + c1ele[0,nsig[0]:] + muexc[0])
            nnew[1,nsig[1]:] = (1-alpha)*n[1,nsig[1]:] + alpha*rhob[1]*np.exp(c1hs[1,nsig[1]:] + c1ele[1,nsig[1]:] + muexc[1])

            n = nnew

            # # var2[0,nsig[0]:] = n[0,nsig[0]:]/np.exp(-Z[0]*psi[nsig[0]:])
            # # var2[1,nsig[1]:] = n[1,nsig[1]:]/np.exp(-Z[1]*psi[nsig[1]:])

            # [varsol2,Omegasol2,Niter] = optimize_fire2(psi,Fpsi2,dFpsidpsi2,n,1.0e-8,0.02,False)

            # # n[0,nsig[0]:] = var2[0,nsig[0]:]*np.exp(-Z[0]*psi[nsig[0]:])
            # # n[1,nsig[1]:] = var2[1,nsig[1]:]*np.exp(-Z[1]*psi[nsig[1]:])

            # psi[:] = varsol2-varsol2[-1]
            psi = sor(n,psi,ele.Z,[Gamma,0],ele.delta,ele.lB,L)

            Omegalast = Omegasol

            Fid = np.sum(n*(np.log(n)-1.0))*delta
            Fhs = np.sum(fmt.Phi(n))*delta
            Fele = ele.free_energy(n,psi)
            Omegasol=(Fid+Fhs+Fele-np.sum(mu[:,np.newaxis]*n*delta)+Gamma*psi[0])/L

            # plt.plot(x,n[0]/rhob[0])
            # plt.plot(x,n[1]/rhob[1])
            # plt.show()

            # plt.plot(x,psi)
            # plt.show()

            error = abs(Omegasol-Omegalast)
            print(error)
            i += 1

        # np.save('profiles-DFTcorr-electrolyte22-c0.5-sigma-0.1704.npy',[x,n[0],n[1],psi])
        np.save('profiles-DFTcorr-Voukadinova2018-electrolyte11-Fig5-Z+=3-rho+=1.0M.npy',[x,n[0],n[1],psi])    

    if test6: 
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