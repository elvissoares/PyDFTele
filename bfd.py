#!/usr/bin/env python3

# This script is the python implementation of the Density Functional Theory
# for Electrolyte Solution in the presence of an external electrostatic potential
#
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2021-06-02
# Updated: 2021-08-24
# Version: 1.1
#
import numpy as np
from scipy.ndimage import convolve1d
from scipy import optimize, interpolate
import matplotlib.pyplot as plt
# from numba import jit
# from multiprocessing import Pool, Process


def phicorr1D(x,b,a):
    ba = b/a
    conds = [(np.abs(x)<=a), (np.abs(x)>a)]
    funcs = [lambda x: (2*np.pi*a**3/3)*(1-3*ba+3*ba**2-3*ba**2*np.abs(x/a)+3*ba*np.abs(x/a)**2-np.abs(x/a)**3),0.0]
    return np.piecewise(x,conds,funcs)

def omegaYK1D(x,kappa,a):
    conds = [(np.abs(x)<=a/2), (np.abs(x)>a/2)]
    funcs = [kappa/(2+kappa*a), lambda x: kappa*np.exp(-kappa*(np.abs(x)-0.5*a))/(2+kappa*a)]
    return np.piecewise(x,conds,funcs)

# The MSA parameters
# @jit(nopython=True)
def Hfunc(rho,a,Gamma):
    return np.sum(rho*a[:,np.newaxis]**3/(1+Gamma*a[:,np.newaxis]),axis=0)+(2.0/np.pi)*(1-(np.pi/6)*np.sum(rho*a[:,np.newaxis]**3,axis=0))

# @jit(nopython=True, parallel=True)
def Etafunc(rho,a,Z,Gamma):
    return np.sum(rho*Z[:,np.newaxis]*a[:,np.newaxis]/(1+Gamma*a[:,np.newaxis]),axis=0)/Hfunc(rho,a,Gamma)

# @jit(nopython=True, parallel=True)
def MSAparameteres(rho,a,Z,lB):
    def fun(x):
        eta = Etafunc(rho,a,Z,x)
        return x**2 - np.pi*lB*np.sum(rho*((Z[:,np.newaxis]-eta*a[:,np.newaxis]**2)/(1+x*a[:,np.newaxis]))**2,axis=0)
    def jac(x):
        eta = Etafunc(rho,a,Z,x)
        h = Hfunc(rho,a,x)
        dhdgamma = -np.sum(rho*a[:,np.newaxis]**4/(1+x*a[:,np.newaxis])**2,axis=0)
        detadgamma = -(eta/h)*dhdgamma-(1/h)*np.sum(rho*Z[:,np.newaxis]*a[:,np.newaxis]**2/(1+x*a[:,np.newaxis])**2,axis=0)
        return np.diag(2*x + 2*np.pi*lB*np.sum(rho*a[:,np.newaxis]*(Z[:,np.newaxis]-eta*a[:,np.newaxis]**2)**2/(1+x*a[:,np.newaxis])**3+rho*a[:,np.newaxis]**2*(Z[:,np.newaxis]-eta*a[:,np.newaxis]**2)*detadgamma/(1+x*a[:,np.newaxis])**2,axis=0))
    kappa = np.sqrt(4*np.pi*lB*np.sum(Z[:,np.newaxis]**2*rho,axis=0))
    amed = np.sum(rho*a[:,np.newaxis],axis=0)/np.sum(rho,axis=0)
    x0 = (np.sqrt(1+2*kappa*amed)-1)/(2*amed)
    sol = optimize.root(fun, x0, jac=jac, method='hybr')
    x = sol.x
    eta = Etafunc(rho,a,Z,x)
    return [x,eta]

def MSAbulkparameteres(rho,a,Z,lB):
    def fun(x):
        Hh = np.sum(rho*a**3/(1+x*a))+(2.0/np.pi)*(1-(np.pi/6)*np.sum(rho*a**3))
        eta = np.sum(rho*Z*a/(1+x*a))/Hh
        return x**2 - np.pi*lB*np.sum(rho*((Z-eta*a**2)/(1+x*a))**2)
    def jac(x):
        h = np.sum(rho*a**3/(1+x*a))+(2.0/np.pi)*(1-(np.pi/6)*np.sum(rho*a**3))
        eta = np.sum(rho*Z*a/(1+x*a))/h
        dhdgamma = -np.sum(rho*a**4/(1+x*a)**2)
        detadgamma = -(eta/h)*dhdgamma-(1/h)*np.sum(rho*Z*a**2/(1+x*a)**2)
        return np.diag(2*x + 2*np.pi*lB*np.sum(rho*a*(Z-eta*a**2)**2/(1+x*a)**3+rho*a**2*(Z-eta*a**2)*detadgamma/(1+x*a)**2))
    kappa = np.sqrt(4*np.pi*lB*np.sum(Z**2*rho))
    amed = np.sum(rho*a)/np.sum(rho)
    x0 = (np.sqrt(1+2*kappa*amed)-1)/(2*amed)
    sol = optimize.root(fun, x0, jac=jac, method='hybr')
    x = sol.x[0]
    h = np.sum(rho*a**3/(1+x*a))+(2.0/np.pi)*(1-(np.pi/6)*np.sum(rho*a**3))
    eta = np.sum(rho*Z*a/(1+x*a))/h
    return [x,eta]

" The DFT model for electrolyte solutions using the generalized grand potential"

class ElectrolyteBFDPlanar():
    def __init__(self,N,delta,species=2,a=np.array([1.0,1.0]),Z=np.array([-1,1]),rhob=np.array([0.1,0.1]),epsr=78.5,model='symmetrical'):
        self.N = N
        self.delta = delta
        self.L = delta*N
        self.a = a
        self.species = species
        self.Z = Z
        self.rhob = rhob
        self.x = np.linspace(0,self.L,N)

        self.half = int(0.5*self.L/self.delta)

        self.lB = 0.714*78.5/epsr # in nm (for water)
        print('lB=',self.lB,' nm')
        
        self.n = np.zeros((self.species,N),dtype=np.float32)
        nphi = int((2*max(self.a))/self.delta+1)
        self.phi = np.zeros((self.species,self.species,nphi),dtype=np.float32)
        self.phiint = np.zeros((self.species,self.species),dtype=np.float32)
        

        [self.Gammabulk,self.Etabulk] = MSAbulkparameteres(self.rhob,self.a,self.Z,self.lB)
        print('inverse Gammabulk = ',1.0/self.Gammabulk,' nm')

        if (model =='symmetrical'):
            self.b = self.a+1.0/self.Gammabulk
            nw = int((max(self.b))/self.delta+1)
            self.w = np.zeros((self.species,nw),dtype=np.float32)
            for i in range(self.species):
                nsig = int(0.5*self.b[i]/self.delta)
                self.w[i,nw//2-nsig:nw//2+nsig] = 1.0/(self.b[i])

                for j in range(self.species):
                    bij = 0.5*(self.b[i]+self.b[j])
                    aij = 0.5*(self.a[i]+self.a[j])
                    nsig = int(aij/self.delta)
                    x = np.linspace(-aij,aij,2*nsig)
                    self.phi[i,j,nphi//2-nsig:nphi//2+nsig] = -(self.Z[i]*self.Z[j]*self.lB/(bij**2))*phicorr1D(x,bij,aij)
                    self.phiint[i,j] = -(self.Z[i]*self.Z[j]*self.lB/(bij**2))*(np.pi*aij**4)*(1-(8/3.0)*bij/aij+2*(bij/aij)**2)

        elif (model =='asymmetrical'):

            X = (self.Z-self.a**2*self.Etabulk)/(1+self.Gammabulk*self.a)
            N = (X-self.Z)/self.a

            self.b = self.a+1.0/self.Gammabulk
            nw = int((max(self.b))/self.delta+1)
            self.w = np.zeros((self.species,nw),dtype=np.float32)

            for i in range(self.species):
                nsig = int(0.5*self.b[i]/self.delta)
                self.w[i,nw//2-nsig:nw//2+nsig] = 1.0/(self.b[i])

                for j in range(self.species):
                    da = 0.5*np.abs(self.a[i]-self.a[j])
                    a = 0.5*(self.a[i]+self.a[j])
                    naij = int(a/self.delta)
                    if (self.a[i]< self.a[j]): A0 = 0.5*da*((X[i]+X[j])*((N[i]+self.Gammabulk*X[i])-(N[j]+self.Gammabulk*X[j]))-0.25*((N[i]+self.Gammabulk*X[i]+N[j]+self.Gammabulk*X[j])**2-4*N[i]*N[j]))
                    else: A0 = 0.5*da*((X[i]+X[j])*((N[j]+self.Gammabulk*X[j])-(N[i]+self.Gammabulk*X[i]))-0.25*((N[i]+self.Gammabulk*X[i]+N[j]+self.Gammabulk*X[j])**2-4*N[i]*N[j]))
                    A1 = -(X[i]-X[j])*(N[i]-N[j])-(X[i]**2+X[j]**2)*self.Gammabulk-2*a*N[i]*N[j]+(1.0/3)*(self.a[i]*(N[i]+self.Gammabulk*X[i])+self.a[j]*(N[j]+self.Gammabulk*X[j]))
                    A2 = (X[i]/self.a[i])*(N[i]+self.Gammabulk*X[i])+(X[j]/self.a[j])*(N[j]+self.Gammabulk*X[j])+N[i]*N[j]-0.5*((N[i]+self.Gammabulk*X[i])**2+(N[j]+self.Gammabulk*X[j])**2)
                    A3 = (1.0/6)*(((N[i]+self.Gammabulk*X[i])/self.a[i])**2+((N[j]+self.Gammabulk*X[j])/self.a[j])**2)
                    if (self.a[i]< self.a[j]): cMSAsh = 2*self.lB*(self.Z[j]*N[j]-X[i]*(N[i]+self.Gammabulk*X[i])+(self.a[i]/3)*(N[i]+self.Gammabulk*X[i])**2)
                    else: cMSAsh = 2*self.lB*(self.Z[i]*N[i]-X[j]*(N[j]+self.Gammabulk*X[j])+(self.a[j]/3)*(N[j]+self.Gammabulk*X[j])**2)
                    x = np.linspace(-a,a,2*naij)
                    self.phi[i,j,nphi//2-naij:nphi//2+naij] = np.piecewise(x,[(np.abs(x)<=da),(np.abs(x)>da)&(np.abs(x)<=a),(np.abs(x)>a)],[lambda x: cMSAsh*np.pi*(x**2-da**2)+2*np.pi*self.lB*self.Z[i]*self.Z[j]*(np.abs(x)-a) - (np.pi*self.lB/15)*(30*A0*(a-da)+15*A1*(a**2-da**2)+10*A2*(a**3-da**3)+6*A3*(a**5-da**5)),lambda x:-(np.pi*self.lB/15)*(30*A0*(a-np.abs(x))+15*A1*(a**2-np.abs(x)**2)+10*A2*(a**3-np.abs(x)**3)+6*A3*(a**5-np.abs(x)**5)) +2*np.pi*self.lB*self.Z[i]*self.Z[j]*(np.abs(x)-a),0.0] )
                    self.phiint[i,j] = -(4*np.pi/3)*cMSAsh*da**3-(np.pi*self.lB/3)*(6*A0*(a**2-da**2)+4*A1*(a**3-da**3)+3*A2*(a**4-da**4)+2*A3*(a**6-da**6))-2*np.pi*a**2*self.lB*self.Z[i]*self.Z[j]
                    # plt.plot(x,self.phi[i,j,nphi//2-naij:nphi//2+naij])
                    # plt.show()

    def Flong(self,psi):
        f = convolve1d(psi, weights=[-1,1], mode='nearest')/self.delta
        return -(1/(8*np.pi*self.lB))*np.sum(f**2)*self.delta

    def Fint(self,rho,psi):
        f2 = np.sum(rho[:,:]*self.Z[:,np.newaxis],axis=0)
        return np.sum(f2*psi)*self.delta 

    def FMSA(self,rho):
        Phi = self.Gammabulk**3/(3*np.pi)
        for i in range(self.species):
            Phi += -self.lB*self.rhob[i]*self.Z[i]*(self.Z[i]*self.Gammabulk+self.Etabulk*self.a[i])/(1+self.Gammabulk*self.a[i])
            dPhieledn = -self.lB*(self.Z[i]**2*self.Gammabulk+2*self.a[i]*self.Etabulk*self.Z[i]-self.Etabulk**2*self.a[i]**3*(2.0/3.0-self.Gammabulk*self.a[i]/3.0))/(1+self.Gammabulk*self.a[i])
            Phi += dPhieledn*(rho[i,:]-self.rhob[i])
            for j in range(self.species):
                Phi += 0.5*(rho[i,:]-self.rhob[i])*convolve1d((rho[j,:]-self.rhob[j]), weights=self.phi[i,j], mode='nearest')*self.delta
        return np.sum(Phi)*self.delta

    def free_energy(self,rho,psi):
        return (self.Flong(psi)+self.Fint(rho,psi)+self.FMSA(rho))

    def c1MSA(self,rho):
        cc = np.zeros((self.species,self.N))
        for i in range(self.species):
            dPhieledn = -self.lB*(self.Z[i]**2*self.Gammabulk+2*self.a[i]*self.Etabulk*self.Z[i]-self.Etabulk**2*self.a[i]**3*(2.0/3.0-self.Gammabulk*self.a[i]/3.0))/(1+self.Gammabulk*self.a[i])
            cc[i,:] = -dPhieledn
            for j in range(self.species):
                cc[i,:] += -convolve1d((rho[j,:]-self.rhob[j]), weights=self.phi[i,j], mode='nearest')*self.delta
        return cc

    def c1long(self,psi):
        cc = np.empty((self.species,self.N))
        for i in range(self.species):
            cc[i,:] = -self.Z[i]*psi[:] 
        return cc

    def c1(self,rho,psi):
        return self.c1MSA(rho)+self.c1long(psi)

    def dOmegadpsi(self,rho,psi,sigma):
        lappsi = (1/(4*np.pi*self.lB))*convolve1d(psi, weights=[1,-2,1], mode='nearest')/self.delta**2
        f = np.zeros_like(psi)
        f[0] = sigma[0]/self.delta
        f[-1] = sigma[1]/self.delta
        f += np.sum(rho[:,:]*self.Z[:,np.newaxis],axis=0) 
        return lappsi + f

    def dOmegadpsi_fixedpotential(self,rho,psi,psi0):
        psi[0] = psi0[0]
        psi[-1] = psi0[1]
        lappsi = (1/(4*np.pi*self.lB))*convolve1d(psi, weights=[1,-2,1], mode='nearest')/self.delta**2
        f = np.zeros_like(psi)
        f += np.sum(rho[:,:]*self.Z[:,np.newaxis],axis=0) 
        return lappsi + f

    def muMSA(self,rhob):
        muu = np.zeros_like(rhob)
        for i in range(self.species):
            muu[i] = -self.lB*(self.Z[i]**2*self.Gammabulk+2*self.a[i]*self.Etabulk*self.Z[i]-self.Etabulk**2*self.a[i]**3*(2.0/3.0-self.Gammabulk*self.a[i]/3.0))/(1+self.Gammabulk*self.a[i])
        return muu

    def mu(self,rhob):
        return self.muMSA(rhob)

if __name__ == "__main__":
    test0 = False # the MSA screening parameter 
    test1 = True # fMSA
    test2 = False # RFD
    test3 = False # fMSA

    import matplotlib.pyplot as plt
    from fire import optimize_fire2
    from fmt import FMTplanar
    from pb import PBplanar

    if test0: 
        c = np.linspace(1e-3,1.0,1000)
        Z = np.array([-1,1])
        a = np.array([0.3,0.3])
        lB = 0.714

        gamma0 = np.zeros_like(c)
        gamma1 = np.zeros_like(c)
        for i in range(c.size):
            rho = np.array([-(Z[1]/Z[0])*c[i],c[i]])*0.622
            kappa = np.sqrt(4*np.pi*lB*np.sum(Z**2*rho))
            # amed = np.power(np.sum(rho*a**3)/np.sum(rho),1.0/3.0)
            amed = np.sqrt(np.sum(rho*a**2)/np.sum(rho))
            gamma0[i] = (np.sqrt(1+2*kappa*amed)-1)/(2*amed)
            [gamma1[i],eta] = MSAbulkparameteres(rho,a,Z,lB)


        plt.plot(c,gamma1,'k')
        plt.plot(c,gamma0,'--',color='grey')
        plt.xlabel('c (mol/L)')
        plt.ylabel('$\Gamma_b$ (nm$^{-1}$)')
        plt.show()
        
    if test1: 
        sigma = np.array([0.3,0.15])
        delta = 0.025*sigma[1]
        L = 4.0 + max(sigma)
        N = int(L/delta)
        Z = np.array([-1,2])

        c = 0.1 #mol/L (equivalent to ionic strength for 1:1)
        rhob = np.array([-(Z[1]/Z[0])*c,c])*6.022e23/1.0e24 # particles/nm^3

        x = np.linspace(0,L,N)

        # Gamma = -0.1704/sigma[0]**2
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
            return -pb.dOmegadpsi(n,psi,[Gamma,0.0])*delta/L

        psi0 = 0.1*Gamma*4*np.pi*pb.lB # attenuation of the surface charge
        psi = np.zeros(N,dtype=np.float32)
        psi[:nsig[0]] = psi0*(1/kD+0.5*sigma[0]-x[:nsig[0]])
        psi[nsig[0]:] = psi0*np.exp(-kD*(x[nsig[0]:]-0.5*sigma[0]))/kD
    
        [varsol,Omegasol,Niter] = optimize_fire2(psi,Fpsi,dFpsidpsi,param,1.0e-8,0.02,False)

        psi[:] = varsol-varsol[-1]

        n[0,:nsig[0]] = 1.0e-16
        n[1,:nsig[1]] = 1.0e-16
        n[0,nsig[0]:] = param[0]*np.exp(-Z[0]*psi[nsig[0]:])
        n[1,nsig[1]:] = param[1]*np.exp(-Z[1]*psi[nsig[1]:])
        # n[0,nsig[0]:] = param[0]
        # n[1,nsig[1]:] = param[1]

        # plt.yscale('log')
        # plt.plot(x,n[0]/rhob[0])
        # plt.plot(x,n[1]/rhob[1],'C3')
        # plt.ylim(1e-1,1e2)
        # plt.show()

        # Now we will solve the DFT with electrostatic correlations
        nn = n.copy()

        fmt = FMTplanar(N,delta,species=2,sigma=sigma)
        ele = ElectrolytefMSA(N,delta,species=2,a=sigma,Z=Z,rhob=rhob,model='symmetrical')

        # ele.auxiliary_quantities(n)
        # plt.plot(x,ele.n[0]/rhob[0])
        # plt.plot(x,ele.n[1]/rhob[1],'C3')
        # plt.show()

        # solving the electrostatic potential equation
        def Fpsi2(psi,nn):
            Fele = ele.Flong(psi)+ele.Fint(nn,psi)
            return (Fele+Gamma*psi[0])/L

        def dFpsidpsi2(psi,nn):
            return -ele.dOmegadpsi(nn,psi,[Gamma,0.0])*delta/L

        mu = np.log(rhob) + fmt.mu(rhob) + ele.mu(rhob)

        # Now we will solve the DFT equations
        def Omega(var,psi):
            nn[0,:] = np.exp(var[0])
            nn[1,:] = np.exp(var[1])
            Fid = np.sum(nn*(var-1.0))*delta
            Fhs = np.sum(fmt.Phi(nn))*delta
            Fele = ele.Flong(psi) + ele.Fint(n,psi) + ele.Fcorr(n)
            return (Fid+Fhs+Fele-np.sum(mu[:,np.newaxis]*nn*delta)+Gamma*psi[0])/L

        def dOmegadnR(var,psi):
            nn[0,:] = np.exp(var[0])
            nn[1,:] = np.exp(var[1])

            [varsol2,Omegasol2,Niter] = optimize_fire2(psi,Fpsi2,dFpsidpsi2,nn,1.0e-8,0.02,False)
            # print(varsol2[-1])
            psi[:] = varsol2-varsol2[-1]

            c1hs = fmt.c1(nn)
            c1ele = ele.c1(nn,psi)
            aux = nn*(var -c1hs -c1ele - mu[:,np.newaxis])*delta/L
            # print(aux[:,-1])
            aux[0,-nsig[0]:] = 0.0
            aux[1,-nsig[1]:] = 0.0
            return aux

        muMSA = ele.muMSA(rhob)
        munonMSA = ele.munonMSA(rhob)
        print('muMSA =',muMSA)
        print('munonMSA =',munonMSA)

        var = np.log(n)
        [varsol,Omegasol1,Niter] = optimize_fire2(var,Omega,dOmegadnR,psi,1.0e-5,0.02,True)
        n[0,:] = np.exp(varsol[0])
        n[1,:] = np.exp(varsol[1])

        c1MSA = ele.c1MSA(n)+muMSA[:,np.newaxis]
        c1nonMSA = ele.c1nonMSA(n)+munonMSA[:,np.newaxis]

        # np.save('profiles-DFTcorr-electrolyte21-c0.5-sigma-0.1704.npy',[x,n[0],n[1],psi,c1MSA[0],c1MSA[1],c1nonMSA[0],c1nonMSA[1]])
        np.save('fMSAdata/profiles-fMSA-Voukadinova2018-electrolyte-Fig3-Z+=2-rho+=0.1M.npy',[x,n[0],n[1],psi,c1MSA[0],c1MSA[1],c1nonMSA[0],c1nonMSA[1]])

    ##################################################################################
    if test2: 
        sigma = np.array([0.425,0.425])
        delta = 0.025*sigma[1]
        L = 10.5*sigma[0]
        N = int(L/delta)
        Z = np.array([-1,2])

        c = 0.5 #mol/L (equivalent to ionic strength for 1:1)
        rhob = np.array([-(Z[1]/Z[0])*c,c])*6.022e23/1.0e24 # particles/nm^3

        x = np.linspace(0,L,N)

        Gamma = -0.1704/sigma[0]**2
        # Gamma = -3.12

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
    
        [varsol,Omegasol,Niter] = optimize_fire2(psi,Fpsi,dFpsidpsi,param,1.0e-8,0.05,False)

        psi[:] = varsol

        n[0,:nsig[0]] = 1.0e-16
        n[1,:nsig[1]] = 1.0e-16
        n[0,nsig[0]:] = param[0]*np.exp(-Z[0]*psi[nsig[0]:])
        n[1,nsig[1]:] = param[1]*np.exp(-Z[1]*psi[nsig[1]:])

        # Now we will solve the DFT with electrostatic correlations
        nn = n.copy()

        fmt = FMTplanar(N,delta,species=2,sigma=sigma)
        ele = ElectrolyteRFD(N,delta,species=2,a=sigma,Z=Z,rhob=rhob)

        # solving the electrostatic potential equation
        def Fpsi2(psi,nn):
            Fele = ele.Flong(nn,psi)+ele.Fint(nn,psi)
            return (Fele+Gamma*psi[0])/L

        def dFpsidpsi2(psi,nn):
            return -ele.dOmegadpsi(nn,psi,[Gamma,0.0])*delta/L

        mu = np.log(rhob) + fmt.mu(rhob) + ele.mu(rhob)

        # Now we will solve the DFT equations
        def Omega(var,psi):
            nn[0,:] = np.exp(var[0])
            nn[1,:] = np.exp(var[1])
            Fid = np.sum(nn*(var-1.0))*delta
            Fhs = np.sum(fmt.Phi(nn))*delta
            Fele = ele.Fint(n,psi) + ele.Fcorr(n)
            return (Fid+Fhs+Fele-np.sum(mu[:,np.newaxis]*nn*delta)+Gamma*psi[0])/L

        def dOmegadnR(var,psi):
            nn[0,:] = np.exp(var[0])
            nn[1,:] = np.exp(var[1])

            [varsol2,Omegasol2,Niter] = optimize_fire2(psi,Fpsi2,dFpsidpsi2,nn,1.0e-6,0.02,False)
            psi[:] = varsol2-varsol2[-1]

            c1hs = fmt.c1(nn)
            c1ele = ele.c1(nn,psi)
            aux = nn*(var -c1hs -c1ele - mu[:,np.newaxis])*delta/L
            # aux[0,-nsig[0]:] = 0.0
            # aux[1,-nsig[1]:] = 0.0
            return aux

        var = np.log(n)
        [varsol,Omegasol1,Niter] = optimize_fire2(var,Omega,dOmegadnR,psi,1.0e-5,0.02,True)
        n[0,:] = np.exp(varsol[0])
        n[1,:] = np.exp(varsol[1])

        muMSA = ele.muMSA(rhob)
        print('muMSA =',muMSA)

        c1MSA = ele.c1MSA(n)+ele.c1nonMSA(n)+muMSA[:,np.newaxis]
        # print(c1MSA)

        np.save('profiles-DFTRFD-electrolyte21-c0.5-sigma-0.1704.npy',[x,n[0],n[1],psi,c1MSA[0],c1MSA[1]])
        # np.save('profiles-DFTcorr-Voukadinova2018-electrolyte-Fig5-Z+=3-rho+=1.0M.npy',[x,n[0],n[1],psi,c1MSA[0],c1MSA[1],c1nonMSA[0],c1nonMSA[1]])

    if test3: 
        sigma = np.array([0.466,0.362])
        delta = 0.025*sigma[1]
        L = 10.5*sigma[0]
        N = int(L/delta)
        Z = np.array([-1,1])

        c = 1.0 #mol/L (equivalent to ionic strength for 1:1)
        rhob = np.array([c,c])*6.022e23/1.0e24 # particles/nm^3

        x = np.linspace(0,L,N)

        # Gamma = -0.1704/sigma[0]**2
        Gamma = 0.1/0.16

        n = np.ones((2,N),dtype=np.float32)
        Vext = np.zeros((2,N),dtype=np.float32)
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
    
        # [varsol,Omegasol,Niter] = optimize_fire2(psi,Fpsi,dFpsidpsi,param,1.0e-8,0.05,False)

        # psi[:] = varsol

        n[0,:nsig[0]] = 1.0e-16
        n[1,:nsig[1]] = 1.0e-16
        n[0,nsig[0]:] = param[0]*np.exp(-Z[0]*psi[nsig[0]:])
        n[1,nsig[1]:] = param[1]*np.exp(-Z[1]*psi[nsig[1]:])

        # Now we will solve the DFT with electrostatic correlations
        nn = n.copy()

        fmt = FMTplanar(N,delta,species=2,sigma=sigma)
        ele = ElectrolytefMSA(N,delta,species=2,a=sigma,Z=Z,rhob=rhob)

        # solving the electrostatic potential equation
        def Fpsi2(psi,nn):
            Fele = ele.Flong(nn,psi)+ele.Fint(nn,psi)
            return (Fele+Gamma*psi[0])/L

        def dFpsidpsi2(psi,nn):
            return -ele.dOmegadpsi(nn,psi,[Gamma,0.0])*delta/L

        mu = np.log(rhob) + fmt.mu(rhob) + ele.mu(rhob)
        # Vext[0,nsig[0]:] = -0.142573/x[nsig[0]:]**3 # SO4
        # Vext[0,nsig[0]:] = -0.0107915/x[nsig[0]:]**3 # I
        # Vext[1,nsig[1]:] = -0.00109374/x[nsig[1]:]**3 # Na

        # Now we will solve the DFT equations
        def Omega(var,psi):
            nn[0,:] = np.exp(var[0])
            nn[1,:] = np.exp(var[1])
            Fid = np.sum(nn*(var-1.0))*delta
            Fhs = np.sum(fmt.Phi(nn))*delta
            Fele = ele.Fint(n,psi) + ele.Fcorr(n)
            return (Fid+Fhs+Fele+np.sum((Vext-mu[:,np.newaxis])*nn)*delta+Gamma*psi[0])/L

        def dOmegadnR(var,psi):
            nn[0,:] = np.exp(var[0])
            nn[1,:] = np.exp(var[1])

            [varsol2,Omegasol2,Niter] = optimize_fire2(psi,Fpsi2,dFpsidpsi2,nn,1.0e-6,0.02,False)
            psi[:] = varsol2-varsol2[-1]

            c1hs = fmt.c1(nn)
            c1ele = ele.c1(nn,psi)
            aux = nn*(var -c1hs -c1ele - mu[:,np.newaxis] + Vext)*delta/L
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

        np.save('profiles-DFTfMSA-Alijo2012-electrolyte-NaI-sigma0.1-nodispersion.npy',[x,n[0],n[1],psi,c1MSA[0],c1MSA[1],c1nonMSA[0],c1nonMSA[1]])