import numpy as np
from scipy.ndimage import convolve1d
from scipy import optimize
import matplotlib.pyplot as plt
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2020-06-02
# Updated: 2021-07-07

def phicorr1D(x,b,a):
    frac = b/a
    conds = [(x<=a/2) & (x>=-a/2), (x>a/2) | (x<-a/2)]
    funcs = [lambda x: (2*np.pi/3)*a**3*(0.125-0.75*frac+1.5*frac**2-3*frac**2*np.abs(x)/a+3*frac*np.abs(x/a)**2-np.abs(x/a)**3),0.0]
    return np.piecewise(x,conds,funcs)

# The MSA parameters
def Hfunc(rho,a,Gamma):
    return np.sum(rho*a**3/(1+Gamma*a))+(2.0/np.pi)*(1-(np.pi/6)*np.sum(rho*a**3))

def Etafunc(rho,a,Z,Gamma):
    return np.sum(rho*Z*a/(1+Gamma*a))/Hfunc(rho,a,Gamma)

def Gammafunc(rho,a,Z,lB):

    def fun(x):
        eta = Etafunc(rho,a,Z,x)
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
    return -lB*(Z**2*Gamma+2*a*eta*Z-eta*a**3*(2.0/3.0-Gamma*a/3.0))/(1+Gamma*a)

# rhob = np.linspace(1e-3,5.0,1000)*0.6
# Z = np.array([-2,1])
# a = np.array([0.452,0.730])
# lB = 0.714

# gamma0 = np.zeros_like(rhob)
# gamma1 = np.zeros_like(rhob)
# for i in range(rhob.size):
#     rho = np.array([rhob[i],2*rhob[i]])
#     kappa = np.sqrt(4*np.pi*lB*np.sum(Z**2*rho))
#     # amed = np.power(np.sum(rho*a**3)/np.sum(rho),1.0/3.0)
#     amed = np.sqrt(np.sum(rho*a**2)/np.sum(rho))
#     gamma0[i] = (np.sqrt(1+2*kappa*amed)-1)/(2*amed)
#     gamma1[i] = Gammafunc(rho,a,Z,lB)


# plt.plot(rhob,gamma1,'k')
# plt.plot(rhob,gamma0,'--',color='grey')
# plt.show()



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

        self.lB = 0.714 # in nm (for water)
        # self.kappa = np.sqrt(4*np.pi*self.lB*np.sum(self.Z**2*self.rhob))
        # print('Debye length: ',1.0/self.kappa,' m')
        
        self.n = np.zeros((self.species,self.N),dtype=np.float32)
        self.phi = np.zeros((self.species,self.species,self.N),dtype=np.float32)
        self.phiint = np.zeros((self.species,self.species),dtype=np.float32)
        self.w = np.zeros((self.species,self.N),dtype=np.float32)
        self.Gamma = np.zeros(self.N,dtype=np.float32)
        self.Eta = np.zeros(self.N,dtype=np.float32)
        self.b = np.zeros(self.species,dtype=np.float32)

        self.Gammabulk = Gammafunc(self.rhob,self.a,self.Z,self.lB)
        self.Etabulk = Etafunc(self.rhob,self.a,self.Z,self.Gammabulk)

        for i in range(self.species):
            self.b[i] = 0.5*self.a[i] + 0.5/self.Gammabulk
            nsig = int(self.b[i]/self.delta)
            self.w[i,self.N//2-nsig:self.N//2+nsig] = 1.0/(2*self.b[i])

            for j in range(i,self.species):
                self.b[j] = 0.5*self.a[j] + 0.5/self.Gammabulk
                nsig = int(0.5*(self.a[i]+self.a[j])/self.delta)
                x = np.linspace(-0.5*(self.a[i]+self.a[j]),(self.a[i]+self.a[j]),2*nsig)
                bij = self.b[i]+self.b[j]
                aij = self.a[i]+self.a[j]
                self.phi[i,j,self.N//2-nsig:self.N//2+nsig] = -self.Z[i]*self.Z[j]*(self.lB/4)*(1/(self.b[i]*self.b[j]))*phicorr1D(x,bij,aij)
                self.phiint[i,j] = -self.Z[i]*self.Z[j]*(self.lB/4)*(1/(self.b[i]*self.b[j]))*(np.pi*aij**3/48)*(3-16*bij/aij+24*(bij/aij)**2)
                self.phiint[j,i] = self.phiint[i,j]

    def auxiliary_quantities(self,rho):
        for i in range(self.species):
            self.n[i,:] = convolve1d(rho[i], weights=self.w[i], mode='nearest')*self.delta
        self.Gamma = np.array([Gammafunc(self.n[:,i],self.a,self.Z,self.lB) for i in range(rho[0].size)])
        self.Eta = np.array([Etafunc(self.n[:,i],self.a,self.Z,self.Gamma[i]) for i in range(rho[0].size)])
        # print(self.Eta.shape)

    def Flong(self,rho,psi):
        f = convolve1d(psi, weights=[-1,1], mode='nearest')/self.delta
        f2 = np.zeros(self.N)
        for i in range(self.species):
            f2 += rho[i,:]*self.Z[i] 
        return -(1/(8*np.pi*self.lB))*np.sum(f**2)*self.delta+ np.sum(f2*psi)*self.delta 

    def Fcorr(self,rho):
        self.auxiliary_quantities(rho)
        Phi = self.Gamma**3/(3*np.pi)
        for i in range(self.species):
            Phi += -self.lB*self.n[i,:]*self.Z[i]*(self.Z[i]*self.Gamma+self.Eta*self.a[i])/(1+self.Gamma*self.a[i])
            for j in range(i,self.species):
                Phi += rho[i,:]*convolve1d(rho[j,:], weights=self.phi[i,j], mode='nearest')*self.delta
        return np.sum(Phi)*self.delta

    def free_energy(self,rho,psi):
        return (self.Flong(rho,psi)+self.Fcorr(rho))

    def c1corr(self,rho):
        self.auxiliary_quantities(rho)
        cc = np.zeros((self.species,self.N))
        for i in range(self.species):
            dPhieledn = -self.lB*(self.Z[i]**2*self.Gamma+2*self.a[i]*self.Eta*self.Z[i]-self.Eta*self.a[i]**3*(2.0/3.0-self.Gamma*self.a[i]/3.0))/(1+self.Gamma*self.a[i])
            cc[i,:] = -convolve1d(dPhieledn, weights=self.w[i], mode='nearest')*self.delta
            for j in range(i,self.species):
                cc[i,:] += -convolve1d(rho[j,:], weights=self.phi[i,j], mode='nearest')*self.delta
        return cc

    def c1long(self,psi):
        cc = np.empty((self.species,self.N))
        for i in range(self.species):
            cc[i,:] = -self.Z[i]*psi[:] 
        return cc

    def c1(self,rho,psi):
        return self.c1corr(rho)+self.c1long(psi)

    def dOmegadpsi(self,rho,psi,sigma):
        lappsi = (1/(4*np.pi*self.lB))*convolve1d(psi, weights=[1,-2,1], mode='nearest')/self.delta**2
        lappsi[0] += sigma[0]/self.delta
        lappsi[-1] += sigma[1]/self.delta
        f = np.zeros(self.N)
        for i in range(self.species):
            f += rho[i,:]*self.Z[i] 
        return lappsi + f

    def mu(self,rhob):
        muu = np.zeros(self.species)
        for i in range(self.species):
            muu[i] = -self.lB*(self.Z[i]**2*self.Gammabulk+2*self.a[i]*self.Etabulk*self.Z[i]-self.Etabulk*self.a[i]**3*(2.0/3.0-self.Gammabulk*self.a[i]/3.0))/(1+self.Gammabulk*self.a[i])
            muu[i] += np.sum(rhob[:]*self.phiint[i,:])

        return muu

if __name__ == "__main__":
    test1 = False #hardwall 
    test2 = False
    test3 = False
    test4 = True

    import matplotlib.pyplot as plt
    import sys
    # sys.path.append("/home/elvis/Documents/Projetos em Andamento/DFT models/PyDFT/")
    from fire import optimize_fire2
    from fmt import FMTplanar

    if test1: 
        sigma = np.array([0.425,0.425])
        delta = 0.01*min(sigma)
        N = 1000
        L = N*delta
        beta = 1.0/40.0
        Z = np.array([-1,1])

        c = 0.1 #mol/L (equivalent to ionic strength for 1:1)
        M2nmunits=6.022e23/1.0e24
        rhob = np.array([c,c])*M2nmunits # particles/nm^3

        fmt = FMTplanar(N,delta,species=2,sigma=sigma)
        ele = Electrolyte(N,delta,species=2,sigma=sigma,Z=Z)
        x = np.linspace(0,L,N)

        Gamma = 0.3/sigma[0]**2
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
            c1hs = fmt.c1(n)
            n[0,:nsig[0]] = 1.0e-16
            n[1,:nsig[1]] = 1.0e-16
            n[0,nsig[0]:] = param[0]*np.exp(-Z[0]*psi[nsig[0]:])
            n[1,nsig[1]:] = param[1]*np.exp(-Z[1]*psi[nsig[1]:])
            Gamma = param[2]
            return -ele.dOmegadpsi(n,psi,Gamma)*delta/L

        param = np.array([rhob[0],rhob[1],Gamma])

        psi0 = Gamma*4*np.pi*ele.lB
        psi = psi0*np.exp(-kappa*(x-sigma[0]/2))
    
        [varsol,Omegasol,Niter] = optimize_fire2(psi,Fpsi,dFpsidpsi,param,1.0e-8,0.02,True)

        psi[:] = varsol

    if test2: 
        sigma = np.array([0.425,0.425])
        delta = 0.02*min(sigma)
        N = 500
        L = N*delta
        beta = 1.0/40.0
        Z = np.array([-1,1])

        c = 1.0 #mol/L (equivalent to ionic strength for 1:1)
        rhob = np.array([c,c])*6.022e23/1.0e24 # particles/nm^3

        fmt = FMTplanar(N,delta,species=2,sigma=sigma)
        ele = Electrolyte(N,delta,species=2,sigma=sigma,Z=Z)
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
        psi = psi0*np.exp(-kD*x)
    
        [varsol,Omegasol,Niter] = optimize_fire2(psi,Fpsi,dFpsidpsi,param,1.0e-8,0.02,False)

        psi[:] = varsol
        # psiPB = psi.copy()
        n[0,:nsig[0]] = 1.0e-16
        n[1,:nsig[1]] = 1.0e-16
        # n[0,nsig[0]:] = rhob[0]
        # n[1,nsig[1]:] = rhob[1]
        n[0,nsig[0]:] = rhob[0]*np.exp(-Z[0]*psi[nsig[0]:])
        n[1,nsig[1]:] = rhob[1]*np.exp(-Z[1]*psi[nsig[1]:])
        # nPB = n.copy()

        # np.save('profiles-PB-electrolyte11-c1.0-sigma0.7.npy',[x,n[0],n[1],psi])

        # Now we will solve the DFT equations
        var = np.array([n[0],n[1],psi])
        aux = np.zeros_like(var)

        def Omega(var,param):
            n[0,:] = var[0]
            n[1,:] = var[1]
            psi[:] = var[2]
            mu[:] = param[0:2]
            Gamma = param[2]
            Fid = np.sum(n*(np.log(var[0:2,:])-1.0))*delta
            Fhs = np.sum(fmt.Phi(n))*delta
            Fele = ele.free_energy(n,psi,beta)
            return (Fid+Fhs+Fele-np.sum(mu[:,np.newaxis]*n)*delta + Gamma*psi[0])/L
        
        def dOmegadnR(var,param):
            n[0,:] = var[0]
            n[1,:] = var[1]
            psi[:] = var[2]
            mu[:] = param[0:2]
            Gamma = param[2]
            c1hs = fmt.c1(n)
            c1ele = ele.c1(n,psi)
            aux[0,:] = (np.log(var[0]) -c1hs[0] -c1ele[0] - mu[0])*delta/L
            aux[1,:] = (np.log(var[1]) -c1hs[1] -c1ele[1] - mu[1])*delta/L
            aux[2,:] = -ele.dOmegadpsi(n,psi,Gamma)*delta/L
            return aux

        mu = np.log(rhob) + fmt.mu(rhob)

        print('mu =',mu)
        print('rhob=',rhob)

        param = np.array([mu[0],mu[1],Gamma])
    
        [varsol,Omegasol,Niter] = optimize_fire2(var,Omega,dOmegadnR,param,1.0e-3,0.0001,True)

        n[0,:] = varsol[0]
        n[1,:] =  varsol[1]
        psi[:] = varsol[2]
        nmean = np.sum(n,axis=1)*delta/L
        print('mu =',mu)
        print('rhob=',rhob,'\n nmean = ',nmean,'\n Omega/L =',Omegasol)

        np.save('profiles-DFTcorr-electrolyte11-c1.0-sigma0.7.npy',[x,n[0],n[1],psi])


    if test3: 
        sigma = np.array([0.425,0.425])
        delta = 0.02*min(sigma)
        N = 500
        L = N*delta
        beta = 1.0/40.0
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
        # psi[:nsig[0]] = psi0*(1/kD+0.5*sigma[0]-x[:nsig[0]])
        # psi[nsig[0]:] = psi0*np.exp(-kD*(x[nsig[0]:]-0.5*sigma[0]))/kD
        psi[:] = psi0*np.exp(-kD*x)
    
        n[0,:nsig[0]] = 1.0e-16
        n[1,:nsig[1]] = 1.0e-16
        n[0,nsig[0]:] = rhob[0]
        n[1,nsig[1]:] = rhob[1]

        nn = n.copy()
        var2 = np.empty_like(n)

        mu = np.log(rhob) + fmt.mu(rhob) + ele.mu(rhob)

        # Now we will solve the DFT equations
        def Omega(var,psi):
            n[0,:] = np.exp(var[0])
            n[1,:] = np.exp(var[1])
            Fid = np.sum(n*(var-1.0))*delta
            Fhs = np.sum(fmt.Phi(n))*delta
            Fele = ele.free_energy(n,psi)
            return (Fid+Fhs+Fele-np.sum(mu[:,np.newaxis]*n*delta))/L
        
        def dOmegadnR(var,psi):
            n[0,:] = np.exp(var[0])
            n[1,:] = np.exp(var[1])
            c1hs = fmt.c1(n)
            c1ele = ele.c1(n,psi)
            return n*(var -c1hs -c1ele - mu[:,np.newaxis])*delta/L

        # solving the regular PB equation
        def Fpsi2(psi,n):
            nn[0,:nsig[0]] = 1.0e-16
            nn[1,:nsig[1]] = 1.0e-16
            nn[0,nsig[0]:] = n[0,nsig[0]:]*np.exp(-Z[0]*psi[nsig[0]:])
            nn[1,nsig[1]:] = n[1,nsig[1]:]*np.exp(-Z[1]*psi[nsig[1]:])
            Fele = ele.free_energy(nn,psi)
            return (Fele+Gamma*psi[0])/L

        def dFpsidpsi2(psi,n):
            nn[0,:nsig[0]] = 1.0e-16
            nn[1,:nsig[1]] = 1.0e-16
            nn[0,nsig[0]:] = n[0,nsig[0]:]*np.exp(-Z[0]*psi[nsig[0]:])
            nn[1,nsig[1]:] = n[1,nsig[1]:]*np.exp(-Z[1]*psi[nsig[1]:])
            return -ele.dOmegadpsi(nn,psi,[Gamma,0])*delta/L

        
        Fid = np.sum(n*(np.log(n)-1.0))*delta
        Fhs = np.sum(fmt.Phi(n))*delta
        Fele = ele.free_energy(n,psi)
        Omegasol=(Fid+Fhs+Fele-np.sum(mu[:,np.newaxis]*n*delta)+Gamma*psi[0])/L
        error = 1.0
        
        while error > 1e-4:

            var = np.log(n)
            [varsol,Omegasol1,Niter] = optimize_fire2(var,Omega,dOmegadnR,psi,1.0e-4,0.02,True)

            n[:] = np.exp(varsol)

            var2[0,nsig[0]:] = n[0,nsig[0]:]/np.exp(-Z[0]*psi[nsig[0]:])
            var2[1,nsig[1]:] = n[1,nsig[1]:]/np.exp(-Z[1]*psi[nsig[1]:])

            [varsol2,Omegasol2,Niter] = optimize_fire2(psi,Fpsi2,dFpsidpsi2,var2,1.0e-4,0.02,False)

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

            error = abs(Omegasol-Omegalast)
            print(error)

        # np.save('profiles-DFTcorr-electrolyte22-c0.5-sigma-0.1704.npy',[x,n[0],n[1],psi])
        np.save('profiles-DFTcorr-electrolyte11-c1.0-sigma-0.7.npy',[x,n[0],n[1],psi])    
        
    if test4: 
        sigma = np.array([0.425,0.425])
        delta = 0.02*min(sigma)
        N = 500
        L = N*delta
        beta = 1.0/40.0
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
            Fele = ele.free_energy(n,psi)
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
        psi = psi0*np.exp(-kD*x)
    
        [varsol,Omegasol,Niter] = optimize_fire2(psi,Fpsi,dFpsidpsi,param,1.0e-8,0.02,False)

        psi[:] = varsol
        n[0,:nsig[0]] = 1.0e-16
        n[1,:nsig[1]] = 1.0e-16
        n[0,nsig[0]:] = rhob[0]*np.exp(-Z[0]*psi[nsig[0]:])
        n[1,nsig[1]:] = rhob[1]*np.exp(-Z[1]*psi[nsig[1]:])

        nn = n.copy()
        var2 = np.empty_like(n)

        mu = np.log(rhob) + fmt.mu(rhob) + ele.mu(rhob)

                # solving the regular PB equation
        def Fpsi2(psi,n):
            nn[0,:nsig[0]] = 1.0e-16
            nn[1,:nsig[1]] = 1.0e-16
            nn[0,nsig[0]:] = n[0,nsig[0]:]*np.exp(-Z[0]*psi[nsig[0]:])
            nn[1,nsig[1]:] = n[1,nsig[1]:]*np.exp(-Z[1]*psi[nsig[1]:])
            Fele = ele.free_energy(nn,psi)
            return (Fele+Gamma*psi[0])/L

        def dFpsidpsi2(psi,n):
            nn[0,:nsig[0]] = 1.0e-16
            nn[1,:nsig[1]] = 1.0e-16
            nn[0,nsig[0]:] = n[0,nsig[0]:]*np.exp(-Z[0]*psi[nsig[0]:])
            nn[1,nsig[1]:] = n[1,nsig[1]:]*np.exp(-Z[1]*psi[nsig[1]:])
            return -ele.dOmegadpsi(nn,psi,[Gamma,0.0])*delta/L

        # Now we will solve the DFT equations
        def Omega(var,psi):
            n[0,:] = np.exp(var[0])
            n[1,:] = np.exp(var[1])
            Fid = np.sum(n*(var-1.0))*delta
            Fhs = np.sum(fmt.Phi(n))*delta
            Fele = ele.free_energy(n,psi)
            return (Fid+Fhs+Fele-np.sum(mu[:,np.newaxis]*n*delta)+Gamma*psi[0])/L
        
        def dOmegadnR(var,psi):
            n[0,:] = np.exp(var[0])
            n[1,:] = np.exp(var[1])

            var2[0,nsig[0]:] = n[0,nsig[0]:]/np.exp(-Z[0]*psi[nsig[0]:])
            var2[1,nsig[1]:] = n[1,nsig[1]:]/np.exp(-Z[1]*psi[nsig[1]:])

            [varsol2,Omegasol2,Niter] = optimize_fire2(psi,Fpsi2,dFpsidpsi2,var2,1.0e-6,0.02,False)

            psi[:] = varsol2

            c1hs = fmt.c1(n)
            c1ele = ele.c1(n,psi)
            return n*(var -c1hs -c1ele - mu[:,np.newaxis])*delta/L

        var = np.log(n)
        [varsol,Omegasol1,Niter] = optimize_fire2(var,Omega,dOmegadnR,psi,1.0e-4,0.02,True)

        n[:] = np.exp(varsol)


        # np.save('profiles-DFTcorr-electrolyte22-c0.5-sigma-0.1704.npy',[x,n[0],n[1],psi])
        np.save('profiles-DFTcorr-electrolyte11-c1.0-sigma-0.7.npy',[x,n[0],n[1],psi])