#!/usr/bin/env python3

# This script is the python implementation of the Density Functional Theory
# for Electrolyte Solution in the presence of an external electrostatic potential
#
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2021-06-02
# Updated: 2022-09-12
# Version: 0.1
#
import numpy as np
import timeit
from scipy.ndimage import convolve1d
from scipy import optimize
from numba import vectorize
from poisson1d import Poisson1D

@vectorize(['f8(f8)','f4(f4)'])
def phi2func(eta):
    if eta < 1.e-3: return 1+eta**2/9
    else: return 1+(2*eta-eta**2+2*np.log(1-eta)*(1-eta))/(3*eta)

@vectorize(['f8(f8)','f4(f4)'])
def phi3func(eta):
    if eta < 1.e-3: return 1-4*eta/9
    else: return 1-(2*eta-3*eta**2+2*eta**3+2*np.log(1-eta)*(1-eta)**2)/(3*eta**2)

@vectorize(['f8(f8)','f4(f4)'])
def phi1func(eta):
    if eta < 1.e-3: return 1-2*eta/9-eta**2/18
    else: return 2*(eta+np.log(1-eta)*(1-eta)**2)/(3*eta**2)

@vectorize(['f8(f8)','f4(f4)'])
def dphi1dnfunc(eta):
    if eta < 1.e-3: 
        return -2/9-eta/9-eta**2/15.0
    else: 
        return (2*(eta-2)*eta+4*(eta-1)*np.log(1-eta))/(3*eta**3)

@vectorize(['f8(f8)','f4(f4)'])
def dphi2dnfunc(eta):
    if eta < 1.e-3: return 2*eta/9+eta**2/6.0
    else: return -(2*eta+eta**2+2*np.log(1-eta))/(3*eta**2)

@vectorize(['f8(f8)','f4(f4)'])
def dphi3dnfunc(eta):
    if eta < 1.e-3: return -4.0/9+eta/9
    else: return -2*(1-eta)*(eta*(2+eta)+2*np.log(1-eta))/(3*eta**3)

def phicorr1D(x,b,a):
    ba = b/a
    conds = [(np.abs(x)<=a), (np.abs(x)>a)]
    funcs = [lambda x: (2*np.pi*a**3/3)*(1-3*ba+3*ba**2-3*ba**2*np.abs(x/a)+3*ba*np.abs(x/a)**2-np.abs(x/a)**3),0.0]
    return np.piecewise(x,conds,funcs)

# The MSA parameters
def Etafunc(rho,a,Z,Gamma):
    H = np.sum(rho*a[:,np.newaxis]**3/(1+Gamma*a[:,np.newaxis]),axis=0)+(2.0/np.pi)*(1-(np.pi/6)*np.sum(rho*a[:,np.newaxis]**3,axis=0))
    return np.sum(rho*Z[:,np.newaxis]*a[:,np.newaxis]/(1+Gamma*a[:,np.newaxis]),axis=0)/H

def Etabulkfunc(rho,a,Z,Gamma):
    H = np.sum(rho*a**3/(1+Gamma*a))+(2.0/np.pi)*(1-(np.pi/6)*np.sum(rho*a**3))
    return np.sum(rho*Z*a/(1+Gamma*a))/H

def Gammabulkparameter(rho,a,Z,lB):
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
    return sol.x[0]

def Gammaparameter(rhoarray,a,Z,lB):
    Gamma = np.empty_like(rhoarray[0])
    for i in range(rhoarray[0].size):
        Gamma[i] = Gammabulkparameter(rhoarray[:,i],a,Z,lB)
    return Gamma

" The DFT model for electrolyte solutions using the generalized grand potential"

" The hard-sphere FMT functional implemented are the following: "
" fmtmethod = RF (Rosenfeld functional) "
"           = WBI (White Bear version I) "
"           = WBII (White Bear version II) "

" The electrostatic correlation functional implemented are the following: "
" ecmethod = PB (Poisson-Boltzmann)  "
"          = MFT (Mean-Field Theory) "
"          = BFD (Bulk fluid density expansion) "
"          = fMSA (functionalized Mean Spherical Approximation symmetrical) "


class ElectrolyteDFT1D():
    def __init__(self,L,lB=0.714,d=np.array([1.0,1.0]),Z=np.array([-1,1]),fmtmethod='WBI',ecmethod='fMSA',geometry='Planar'):
        self.geometry = geometry
        self.fmtmethod = fmtmethod
        self.ecmethod = ecmethod
        self.L = L
        self.d = d
        self.delta = 0.01*self.d.min()
        self.species = d.size
        self.Z = Z
        self.z = np.arange(0,self.L,self.delta)+0.5*self.delta
        self.N = self.z.size

        self.rho = np.empty((self.species,self.N),dtype=np.float32)
        self.c1 = np.empty((self.species,self.N),dtype=np.float32)
        self.c1exc = np.empty((self.species,self.N),dtype=np.float32)
        self.Vext = np.zeros((self.species,self.N),dtype=np.float32)

        # defining the FMT terms
        self.n3 = np.empty(self.N,dtype=np.float32)
        self.n2 = np.empty(self.N,dtype=np.float32)
        self.n2vec = np.empty(self.N,dtype=np.float32)

        self.w3 = np.empty(self.species,dtype=object)
        self.w2 = np.empty(self.species,dtype=object)
        self.w2vec = np.empty(self.species,dtype=object)
        self.c1hs = np.empty((self.species,self.N),dtype=np.float32)

        for i in range(self.species):
            nd = int(self.d[i]/self.delta)+1
            x = np.linspace(-0.5*self.d[i],0.5*self.d[i],nd)
            self.w3[i] = np.pi*((0.5*self.d[i])**2-x**2)
            self.w2[i] = self.d[i]*np.pi*np.ones(nd)
            self.w2vec[i] = 2*np.pi*x

        # defining the EC terms
        self.lB = lB # in nm
        self.phi = np.zeros((self.species,self.species),dtype=object)
        self.phiint = np.zeros((self.species,self.species),dtype=np.float32)
        self.c1long = np.zeros((self.species,self.N),dtype=np.float32)
        self.c1ec = np.zeros((self.species,self.N),dtype=np.float32)
        
        self.psi = np.zeros(self.N,dtype=np.float32)

        print('==== The DFT for electrolyte system ====')
        print('Geometry:',self.geometry )
        print('FMT method:',self.fmtmethod )
        print('EC method:',self.ecmethod )
        print('L =',self.L)
        print('---------------------------')
        print('Number os species =',self.species)
        print('d:',self.d)

    def Set_BulkDensities(self,rhob):

        print('---- Setting bulk quantities ----')

        self.rhob = rhob

        self.kD = np.sqrt(4*np.pi*self.lB*np.sum(self.Z**2*self.rhob))

        self.Gammabulk = Gammabulkparameter(self.rhob,self.d,self.Z,self.lB)
        self.Etabulk = Etabulkfunc(self.rhob,self.d,self.Z,self.Gammabulk)
        self.b = self.d+1.0/self.Gammabulk

        if self.ecmethod == 'fMSA':

            self.ws = np.empty(self.species,dtype=object)
            self.q0 = np.zeros((self.species,self.N),dtype=np.float32)

            self.Gamma = np.zeros(self.N,dtype=np.float32)
            self.Eta = np.zeros(self.N,dtype=np.float32)
            
            for i in range(self.species):
                nd = int(self.b[i]/self.delta)+1
                self.ws[i] = np.ones(nd)/(self.b[i])

                for j in range(self.species):
                    bij = 0.5*(self.b[i]+self.b[j])
                    aij = 0.5*(self.d[i]+self.d[j])
                    nsig = int(aij/self.delta)
                    x = np.linspace(-aij,aij,2*nsig)
                    self.phi[i,j] = -(self.Z[i]*self.Z[j]*self.lB/(bij**2))*phicorr1D(x,bij,aij)
                    self.phiint[i,j] = -(self.Z[i]*self.Z[j]*self.lB/(bij**2))*(np.pi*aij**4)*(1-(8/3.0)*bij/aij+2*(bij/aij)**2)

        elif self.ecmethod == 'BFD':

            X = (self.Z-self.d**2*self.Etabulk)/(1+self.Gammabulk*self.d)
            N = (X-self.Z)/self.d

            for i in range(self.species):
                nd = int(self.b[i]/self.delta)+1
                # self.ws[i] = np.ones(nd)/(self.b[i])

                for j in range(self.species):
                    da = 0.5*np.abs(self.d[i]-self.d[j])
                    a = 0.5*(self.d[i]+self.d[j])
                    naij = int(a/self.delta)
                    if (self.d[i]< self.d[j]): A0 = 0.5*da*((X[i]+X[j])*((N[i]+self.Gammabulk*X[i])-(N[j]+self.Gammabulk*X[j]))-0.25*((N[i]+self.Gammabulk*X[i]+N[j]+self.Gammabulk*X[j])**2-4*N[i]*N[j]))
                    else: A0 = 0.5*da*((X[i]+X[j])*((N[j]+self.Gammabulk*X[j])-(N[i]+self.Gammabulk*X[i]))-0.25*((N[i]+self.Gammabulk*X[i]+N[j]+self.Gammabulk*X[j])**2-4*N[i]*N[j]))
                    A1 = -(X[i]-X[j])*(N[i]-N[j])-(X[i]**2+X[j]**2)*self.Gammabulk-2*a*N[i]*N[j]+(1.0/3)*(self.d[i]*(N[i]+self.Gammabulk*X[i])+self.d[j]*(N[j]+self.Gammabulk*X[j]))
                    A2 = (X[i]/self.d[i])*(N[i]+self.Gammabulk*X[i])+(X[j]/self.d[j])*(N[j]+self.Gammabulk*X[j])+N[i]*N[j]-0.5*((N[i]+self.Gammabulk*X[i])**2+(N[j]+self.Gammabulk*X[j])**2)
                    A3 = (1.0/6)*(((N[i]+self.Gammabulk*X[i])/self.d[i])**2+((N[j]+self.Gammabulk*X[j])/self.d[j])**2)
                    if (self.d[i]< self.d[j]): cMSAsh = 2*self.lB*(self.Z[j]*N[j]-X[i]*(N[i]+self.Gammabulk*X[i])+(self.d[i]/3)*(N[i]+self.Gammabulk*X[i])**2)
                    else: cMSAsh = 2*self.lB*(self.Z[i]*N[i]-X[j]*(N[j]+self.Gammabulk*X[j])+(self.d[j]/3)*(N[j]+self.Gammabulk*X[j])**2)
                    x = np.linspace(-a,a,2*naij)
                    self.phi[i,j] = np.piecewise(x,[(np.abs(x)<=da),(np.abs(x)>da)&(np.abs(x)<=a),(np.abs(x)>a)],[lambda x: cMSAsh*np.pi*(x**2-da**2)+2*np.pi*self.lB*self.Z[i]*self.Z[j]*(np.abs(x)-a) - (np.pi*self.lB/15)*(30*A0*(a-da)+15*A1*(a**2-da**2)+10*A2*(a**3-da**3)+6*A3*(a**5-da**5)),lambda x:-(np.pi*self.lB/15)*(30*A0*(a-np.abs(x))+15*A1*(a**2-np.abs(x)**2)+10*A2*(a**3-np.abs(x)**3)+6*A3*(a**5-np.abs(x)**5)) +2*np.pi*self.lB*self.Z[i]*self.Z[j]*(np.abs(x)-a),0.0] )
                    self.phiint[i,j] = -(4*np.pi/3)*cMSAsh*da**3-(np.pi*self.lB/3)*(6*A0*(a**2-da**2)+4*A1*(a**3-da**3)+3*A2*(a**4-da**4)+2*A3*(a**6-da**6))-2*np.pi*a**2*self.lB*self.Z[i]*self.Z[j]

        self.Calculate_mu()
        print('rhob:',self.rhob)
        print('muid:',self.muid)
        print('muhs:',self.muhs)
        print('muec:',self.muec)

    def Update_System(self):
        if self.ecmethod != 'PB': self.Calculate_auxiliary_quantities()
        self.Calculate_Potential()
        self.Calculate_c1()
        self.Calculate_Omega()

    def Set_Boundary_Conditions(self,params=np.array([0.0,0.0]),bc='potential'):
        print('---- Setting electrostatic boundary conditions ----')
        self.bc = bc
        if self.bc == 'potential':
            self.psi0 = params
            self.psi[0] = self.psi0[0]
        elif self.bc == 'sigma':
            self.sigma = params
            self.poisson = Poisson1D(self.N,self.delta,lB=self.lB,boundary_condition='mixed')
        print('Boundary condition is:',self.bc)
        print('Values =',params)

    def Set_InitialCondition(self):
        nsig = (0.5*self.d/self.delta).astype(int)
        n2sig = (self.d/self.delta).astype(int)
        if self.extpotmodel  == 'hardwall':
            for i in range(self.species):
                self.rho[i,:] = self.rhob[i]
                self.rho[i,:nsig[i]] = 1.0e-16
        elif self.extpotmodel  == 'hardpore':
            for i in range(self.species):
                self.rho[i,:] = self.rhob[i]
                self.rho[i,:nsig[i]] = 1.0e-16
                self.rho[i,-nsig[i]:] = 1.0e-16
        elif self.extpotmodel  == 'hardsphere':
            for i in range(self.species):
                self.rho[i,:] = self.rhob[i]
                self.rho[i,:n2sig[i]] = 1.0e-16

    def Set_External_Potential(self,extpotmodel='hardwall',params='None'):
        print('---- Setting nonelectrostatic external potential ----')
        self.extpotmodel = extpotmodel
        self.params = params
        if self.extpotmodel  == 'hardwall' or self.extpotmodel  == 'hardpore' or self.extpotmodel  == 'hardsphere':
            self.Vext[:] = 0.0
        print('External Potential model is:',self.extpotmodel)

    def Calculate_auxiliary_quantities(self):
        self.n3[:] = convolve1d(self.rho[0], weights=self.w3[0], mode='nearest')*self.delta
        self.n2[:] = convolve1d(self.rho[0], weights=self.w2[0], mode='nearest')*self.delta
        self.n2vec[:] = convolve1d(self.rho[0], weights=self.w2vec[0], mode='nearest')*self.delta
        self.n1vec = self.n2vec/(2*np.pi*self.d[0])
        self.n0 = self.n2/(np.pi*self.d[0]**2)
        self.n1 = self.n2/(2*np.pi*self.d[0])
        
        for i in range(1,self.species):
            self.n3[:] += convolve1d(self.rho[i], weights=self.w3[i], mode='nearest')*self.delta
            n2 = convolve1d(self.rho[i], weights=self.w2[i], mode='nearest')*self.delta
            n2vec = convolve1d(self.rho[i], weights=self.w2vec[i], mode='nearest')*self.delta
            self.n2[:] += n2
            self.n2vec[:] += n2vec
            self.n1vec[:] += n2vec/(2*np.pi*self.d[i])
            self.n0[:] += n2/(np.pi*self.d[i]**2)
            self.n1[:] += n2/(2*np.pi*self.d[i])

        if self.ecmethod == 'fMSA':
            for i in range(self.species):
                self.q0[i,:] = convolve1d(self.rho[i], weights=self.ws[i], mode='nearest')*self.delta
            self.Gamma[:] = Gammaparameter(self.q0,self.d,self.Z,self.lB) 
            self.Eta[:] = Etafunc(self.q0,self.d,self.Z,self.Gamma)
            #verified

        self.oneminusn3 = 1-self.n3

        if self.fmtmethod == 'RF' or self.fmtmethod == 'WBI': 
            self.phi2 = 1.0
            self.dphi2dn3 = 0.0
        elif self.fmtmethod == 'WBII': 
            self.phi2 = phi2func(self.n3)
            self.dphi2dn3 = dphi2dnfunc(self.n3)

        if self.fmtmethod == 'WBI': 
            self.phi3 = phi1func(self.n3)
            self.dphi3dn3 = dphi1dnfunc(self.n3)
        elif self.fmtmethod == 'WBII': 
            self.phi3 = phi3func(self.n3)
            self.dphi3dn3 = dphi3dnfunc(self.n3)
        else:
            self.phi3 = 1.0
            self.dphi3dn3 = 0.0

    def Calculate_Free_energy(self):
        self.Fid = np.sum(self.rho*(np.log(self.rho)-1.0))*self.delta

        aux = convolve1d(self.psi, weights=[-1,1], mode='nearest')/self.delta
        self.Fcoul = -(1/(8*np.pi*self.lB))*np.sum(aux**2)*self.delta

        aux = np.sum(self.rho[:,:]*self.Z[:,np.newaxis],axis=0)
        self.Fcoul += np.sum(aux*self.psi)*self.delta 

        if self.ecmethod == 'PB':
            self.Fec = 0.0
            self.Fhs = 0.0
        else:
            aux = -self.n0*np.log(self.oneminusn3)+(self.phi2/self.oneminusn3)*(self.n1*self.n2-(self.n1vec*self.n2vec)) + (self.phi3/(24*np.pi*self.oneminusn3**2))*(self.n2*self.n2*self.n2-3*self.n2*(self.n2vec*self.n2vec))
            self.Fhs = np.sum(aux)*self.delta

            if self.ecmethod == 'MFT':
                self.Fec = 0.0
            elif self.ecmethod == 'fMSA':
                aux = self.Gamma**3/(3*np.pi)
                for i in range(self.species):
                    aux[:] += -self.lB*self.q0[i,:]*self.Z[i]*(self.Z[i]*self.Gamma+self.Eta*self.d[i])/(1+self.Gamma*self.d[i])
                    for j in range(self.species):
                        aux[:] += 0.5*(self.rho[i,:]-self.rhob[i])*convolve1d((self.rho[j,:]-self.rhob[j]), weights=self.phi[i,j], mode='nearest')*self.delta
                self.Fec = np.sum(aux)*self.delta
            elif self.ecmethod == 'BFD':
                aux = np.ones_like(self.z)*(self.Gammabulk**3/(3*np.pi) - np.sum(self.lB*self.rhob*self.Z*(self.Z*self.Gammabulk+self.Etabulk*self.d)/(1+self.Gammabulk*self.d))) 
                for i in range(self.species):
                    aux[:] += -(self.lB*(self.Z[i]**2*self.Gammabulk+2*self.d[i]*self.Etabulk*self.Z[i]-self.Etabulk**2*self.d[i]**3*(2.0/3.0-self.Gammabulk*self.d[i]/3.0))/(1+self.Gammabulk*self.d[i]))*(self.rho[i,:]-self.rhob[i])
                    for j in range(self.species):
                        aux[:] += 0.5*(self.rho[i,:]-self.rhob[i])*convolve1d((self.rho[j,:]-self.rhob[j]), weights=self.phi[i,j], mode='nearest')*self.delta
                self.Fec = np.sum(aux)*self.delta
        self.Fexc = self.Fhs+self.Fec
        self.F = self.Fid+self.Fexc+self.Fcoul

    def Calculate_Omega(self):
        self.Calculate_Free_energy()
        self.Omega = (self.F + np.sum((self.Vext-self.mu[:,np.newaxis])*self.rho)*self.delta)/self.L

    def Calculate_Potential(self):
        self.q = np.sum(self.rho*self.Z[:,np.newaxis],axis=0)
        if self.bc == 'potential':
            Q = np.cumsum(self.q[::-1])
            if self.psi0[-1] == 0.0:
                self.psi[:] = self.psi0[0] + (4*np.pi*self.lB)*self.z*Q[::-1]*self.delta +(4*np.pi*self.lB)*np.cumsum(self.z*self.q)*self.delta
            self.sigma = -np.sum(self.q)*self.delta
        elif self.bc == 'sigma':
            if self.sigma[-1] == 0.0:
                self.psi[:] = self.poisson.ElectrostaticPotential(self.q,self.sigma)

    def Calculate_c1(self):
        if self.ecmethod == 'PB':
            self.c1ec[:,:] = 0.0
            self.c1hs[:,:] = 0.0
            self.c1exc[:,:] = 0.0
        else:
            dPhidn0 = -np.log(self.oneminusn3 )
            dPhidn1 = self.n2*self.phi2/self.oneminusn3
            dPhidn2 = self.n1*self.phi2/self.oneminusn3  + (3*self.n2*self.n2-3*(self.n2vec*self.n2vec))*self.phi3/(24*np.pi*self.oneminusn3**2)

            dPhidn3 = self.n0/self.oneminusn3 +(self.n1*self.n2-(self.n1vec*self.n2vec))*(self.dphi2dn3 + self.phi2/self.oneminusn3)/self.oneminusn3 + (self.n2*self.n2*self.n2-3*self.n2*(self.n2vec*self.n2vec))*(self.dphi3dn3+2*self.phi3/self.oneminusn3)/(24*np.pi*self.oneminusn3**2)

            dPhidn1vec0 = -self.n2vec*self.phi2/self.oneminusn3 
            dPhidn2vec0 = -self.n1vec*self.phi2/self.oneminusn3  - self.n2*self.n2vec*self.phi3/(4*np.pi*self.oneminusn3**2)

            # EC c1
            for i in range(self.species):
                self.c1hs[i,:] = -convolve1d(dPhidn2 + dPhidn1/(2*np.pi*self.d[i]) + dPhidn0/(np.pi*self.d[i]**2), weights=self.w2[i], mode='nearest')*self.delta - convolve1d(dPhidn3, weights=self.w3[i], mode='nearest')*self.delta + convolve1d(dPhidn2vec0+dPhidn1vec0/(2*np.pi*self.d[i]), weights=self.w2vec[i], mode='nearest')*self.delta

            del dPhidn0,dPhidn1,dPhidn2,dPhidn3,dPhidn1vec0,dPhidn2vec0

            if self.ecmethod == 'MFT':
                self.c1ec[:,:] = 0.0
            elif self.ecmethod == 'BFD':
                for i in range(self.species):
                    self.c1ec[i,:] = -self.lB*(self.Z[i]**2*self.Gammabulk+2*self.d[i]*self.Etabulk*self.Z[i]-self.Etabulk**2*self.d[i]**3*(2.0/3.0-self.Gammabulk*self.d[i]/3.0))/(1+self.Gammabulk*self.d[i])
                    for j in range(self.species):
                        self.c1ec[i,:] += -convolve1d((self.rho[j,:]-self.rhob[j]), weights=self.phi[i,j], mode='nearest')*self.delta
            elif self.ecmethod == 'fMSA':
                for i in range(self.species):
                    dPhieledn = -self.lB*(self.Z[i]**2*self.Gamma+2*self.d[i]*self.Eta*self.Z[i]-self.Eta**2*self.d[i]**3*(2.0/3.0-self.Gamma*self.d[i]/3.0))/(1+self.Gamma*self.d[i])
                    self.c1ec[i,:] = -convolve1d(dPhieledn, weights=self.ws[i], mode='nearest')*self.delta
                    del dPhieledn
                    for j in range(self.species):
                        self.c1ec[i,:] += -convolve1d((self.rho[j,:]-self.rhob[j]), weights=self.phi[i,j], mode='nearest')*self.delta

            self.c1exc = self.c1hs+self.c1ec
        
        # Coulomb c1
        self.c1coul = -self.Z[:,np.newaxis]*self.psi

        self.c1 = self.c1exc+self.c1coul

    def Calculate_mu(self):
        self.muid = np.log(self.rhob)

        if self.ecmethod == 'PB':
            self.muhs = np.zeros_like(self.rhob)
            self.muec = np.zeros_like(self.rhob)
            self.muexc = np.zeros_like(self.rhob)
        else:
            # HS chemical potential
            n3 = np.sum(self.rhob*np.pi*self.d**3/6)
            n2 = np.sum(self.rhob*np.pi*self.d**2)
            n1 = np.sum(self.rhob*self.d/2)
            n0 = np.sum(self.rhob)
            oneminusn3 = 1- n3

            if self.fmtmethod == 'RF' or self.fmtmethod == 'WBI': 
                phi2 = 1.0
                dphi2dn3 = 0.0
            elif self.fmtmethod == 'WBII': 
                phi2 = phi2func(n3)
                dphi2dn3 = dphi2dnfunc(n3)

            if self.fmtmethod == 'WBI': 
                phi3 = phi1func(n3)
                dphi3dn3 = dphi1dnfunc(n3)
            elif self.fmtmethod == 'WBII': 
                phi3 = phi3func(n3)
                dphi3dn3 = dphi3dnfunc(n3)
            else: 
                phi3 = 1.0
                dphi3dn3 = 0.0

            dPhidn0 = -np.log(oneminusn3 )
            dPhidn1 = n2*phi2/oneminusn3
            dPhidn2 = n1*phi2/oneminusn3  + (3*n2*n2)*phi3/(24*np.pi*oneminusn3**2)
            dPhidn3 = n0/oneminusn3 +(n1*n2)*(dphi2dn3 + phi2/oneminusn3)/oneminusn3 + (n2*n2*n2)*(dphi3dn3+2*phi3/oneminusn3)/(24*np.pi*oneminusn3**2)

            self.muhs = dPhidn0+dPhidn1*self.d/2+dPhidn2*np.pi*self.d**2+dPhidn3*np.pi*self.d**3/6

            # EC chemical potential
            if self.ecmethod == 'MFT':
                self.muec = np.zeros_like(self.Z)
            else:
                self.muec = self.lB*(self.Z**2*self.Gammabulk+2*self.d*self.Etabulk*self.Z-self.Etabulk**2*self.d**3*(2.0/3.0-self.Gammabulk*self.d/3.0))/(1+self.Gammabulk*self.d)

            self.muexc = self.muhs+self.muec

        self.mu = self.muid + self.muexc

    def Calculate_Equilibrium(self,method='fire',alpha0=0.19,dt=0.01,rtol=1e-4,atol=1e-5,logoutput=False):

        print('---- Obtaining the thermodynamic equilibrium ----')

        starttime = timeit.default_timer()

        lnrho = np.log(self.rho)

        if method == 'picard':
            nsig = (0.5*self.d/self.delta).astype(int)  
            errorlast = np.inf

            for i in range(Nmax):
                lnrhonew = self.c1 + self.mu[:,np.newaxis] - self.Vext
                for k in range(self.species):
                    lnrhonew[k,:nsig[k]] = np.log(1.0e-16)
                
                lnrho[:] = (1-alpha)*lnrho + alpha*lnrhonew
                self.rho[:] = np.exp(lnrho)
                self.Update_System()

                F = (lnrho - lnrhonew)
                error = np.linalg.norm(F/(atol+rtol*np.abs(lnrho)))

                if errorlast > error:  alpha = min(0.02,alpha*finc)
                else: alpha = max(1.0e-3,alpha*fdec)

                errorlast = error
                if error < 1.0: break
                if logoutput: print(i,self.Omega,error)
            self.Niter = i

        elif method == 'fire':

            # Fire algorithm
            Ndelay = 20
            Nmax = 10000
            finc = 1.1
            fdec = 0.5
            fa = 0.99
            Nnegmax = 2000
            dtmax = 10*dt
            dtmin = 0.02*dt
            alpha = alpha0
            Npos = 0
            Nneg = 0

            V = np.zeros_like(self.rho)
            F = -self.rho*(lnrho -self.c1 - self.mu[:,np.newaxis]+self.Vext)

            error0 = max(np.abs(F.min()),F.max())

            for i in range(Nmax):

                P = (F*V).sum() # dissipated power
                if (P>0):
                    Npos = Npos + 1
                    if Npos>Ndelay:
                        dt = min(dt*finc,dtmax)
                        alpha = alpha*fa
                else:
                    Npos = 0
                    Nneg = Nneg + 1
                    if Nneg > Nnegmax: break
                    if i> Ndelay:
                        dt = max(dt*fdec,dtmin)
                        alpha = alpha0
                    lnrho[:] += - 0.5*dt*V
                    V[:] = 0.0
                    self.rho[:] = np.exp(lnrho)
                    self.Update_System()

                V[:] += 0.5*dt*F
                V[:] = (1-alpha)*V + alpha*F*np.linalg.norm(V)/np.linalg.norm(F)
                lnrho[:] += dt*V
                self.rho[:] = np.exp(lnrho)
                self.Update_System()
                F[:] = -self.rho*(lnrho -self.c1 - self.mu[:,np.newaxis]+self.Vext)
                V[:] += 0.5*dt*F

                error = max(np.abs(F.min()),F.max())
                if error/error0 < rtol and error < atol: break

                if logoutput: print(i,self.Omega,error)
            self.Niter = i

            del V, F  

        print("Time to achieve equilibrium:", timeit.default_timer() - starttime, 'sec')
        print('Number of iterations:', self.Niter)
        print('error:', error)
        print('---- Equilibrium quantities ----')
        print('Fid =',self.Fid)
        print('Fexc =',self.Fexc)
        print('Fcoul =',self.Fcoul)
        print('Omega =',self.Omega)
        print('================================')