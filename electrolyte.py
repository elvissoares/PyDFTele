#!/usr/bin/env python3

# This script is the python implementation of the Density Functional Theory
# for Electrolyte Solution in the presence of an external electrostatic potential
#
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2021-06-02
# Updated: 2022-01-17
# Version: 2.0
#
import numpy as np
from scipy.ndimage import convolve1d
from scipy import optimize
from scipy.linalg import solve_banded
from numba import jit, njit, vectorize, prange, int32, float32, float64    # import the types
from numba.experimental import jitclass
import matplotlib.pyplot as plt
from poisson1d import Poisson1D
from optimizer import Optimize

" Global variables for the FIRE algorithm"
Ndelay = 20
Nmax = 10000
finc = 1.1
fdec = 0.5
fa = 0.99
Nnegmax = 2000

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
spec = [
    ('N', int32),   
    ('delta', float32),  
    ('L', float32),   
    ('species', int32), 
    ('d', float32[:]), 
    ('Z', float32[:]), 
    ('rhob', float32[:]), 
    ('x', float32[:]), 
    ('rho', float32[:,:]), 
]

# @jitclass(spec)
class ElectrolyteDFT():
    def __init__(self,N,delta,lB=0.714,d=np.array([1.0,1.0]),Z=np.array([-1,1]),rhob=np.array([0.1,0.1]),fmtmethod='WBI',ecmethod='fMSA'):
        self.fmtmethod = fmtmethod
        self.ecmethod = ecmethod
        self.N = N
        self.delta = delta
        self.L = delta*N
        self.d = d
        self.species = d.size
        self.Z = Z
        self.rhob = rhob
        self.x = np.linspace(0,self.L,N)

        self.rho = np.empty((self.species,self.N),dtype=np.float32)
        self.c1 = np.zeros((self.species,self.N),dtype=np.float32)
        self.c1exc = np.zeros((self.species,self.N),dtype=np.float32)
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
        self.q0 = np.zeros((self.species,self.N),dtype=np.float32)
        self.phi = np.zeros((self.species,self.species),dtype=object)
        self.phiint = np.zeros((self.species,self.species),dtype=np.float32)
        self.c1long = np.zeros((self.species,self.N),dtype=np.float32)
        self.c1ec = np.zeros((self.species,self.N),dtype=np.float32)
        self.ws = np.zeros(self.species,dtype=object)
        
        self.Gamma = np.zeros(N,dtype=np.float32)
        self.Eta = np.zeros(N,dtype=np.float32)
        self.psi = np.zeros(N,dtype=np.float32)

        # self.poisson = Poisson1D(self.N,self.delta,boundary_condition='dirichlet')
        self.Gammabulk = Gammabulkparameter(self.rhob,self.d,self.Z,self.lB)
        self.Etabulk = Etafunc(self.rhob,self.d,self.Z,self.Gammabulk)
        self.b = self.d+1.0/self.Gammabulk
        print('b = ',self.b,' nm')

        if self.ecmethod == 'fMSA':
            
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

        elif self.ecmethod == 'fMSA-asymmetrical' or self.ecmethod == 'BFD':

            X = (self.Z-self.d**2*self.Etabulk)/(1+self.Gammabulk*self.d)
            N = (X-self.Z)/self.d

            for i in range(self.species):
                nd = int(self.b[i]/self.delta)+1
                self.ws[i] = np.ones(nd)/(self.b[i])

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
        self.Set_External_Potential()

    def Update_System(self):
        self.Calculate_auxiliary_quantities()
        self.Calculate_Potential()
        self.Calculate_c1()
        self.Calculate_Omega()

    def Set_InitialCondition(self):
        nsig = (0.5*self.d/self.delta).astype(int)
        self.rho[0,:] = self.rhob[0]
        self.rho[1,:] = self.rhob[1]
        self.rho[0,:nsig[0]] = 1.0e-16
        self.rho[1,:nsig[1]] = 1.0e-16

    def Set_External_Potential(self,Vext=0.0):
        self.Vext[:] = Vext

    def Set_Boundary_Conditions(self,psi0=0.0,sigma=np.array([0.0,0.0])):
        self.psi0 = psi0
        if self.psi0 == 0.0: self.sigma = sigma

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

        if self.ecmethod == 'fMSA' or self.ecmethod == 'fMSA-asymmetrical':
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

        aux = -self.n0*np.log(self.oneminusn3)+(self.phi2/self.oneminusn3)*(self.n1*self.n2-(self.n1vec*self.n2vec)) + (self.phi3/(24*np.pi*self.oneminusn3**2))*(self.n2*self.n2*self.n2-3*self.n2*(self.n2vec*self.n2vec))
        self.Fhs = np.sum(aux)*self.delta

        aux = convolve1d(self.psi, weights=[-1,1], mode='nearest')/self.delta
        self.Fcoul = -(1/(8*np.pi*self.lB))*np.sum(aux**2)*self.delta

        aux = np.sum(self.rho[:,:]*self.Z[:,np.newaxis],axis=0)
        self.Fcoul += np.sum(aux*self.psi)*self.delta 

        if self.ecmethod == 'PB':
            self.Fec = 0.0
        elif self.ecmethod == 'fMSA' or self.ecmethod == 'fMSA-asymmetrical':
            aux = self.Gamma**3/(3*np.pi)
            for i in range(self.species):
                aux += -self.lB*self.q0[i,:]*self.Z[i]*(self.Z[i]*self.Gamma+self.Eta*self.d[i])/(1+self.Gamma*self.d[i])
                for j in range(self.species):
                    aux += 0.5*(self.rho[i,:]-self.rhob[i])*convolve1d((self.rho[j,:]-self.rhob[j]), weights=self.phi[i,j], mode='nearest')*self.delta
            self.Fec = np.sum(aux)*self.delta
        elif self.ecmethod == 'BFD':
            aux = np.ones_like(self.x)*(self.Gammabulk**3/(3*np.pi) - np.sum(self.lB*self.rhob*self.Z*(self.Z*self.Gammabulk+self.Etabulk*self.d)/(1+self.Gammabulk*self.d)))
            for i in range(self.species):
                for j in range(self.species):
                    aux += 0.5*(self.rho[i,:]-self.rhob[i])*convolve1d((self.rho[j,:]-self.rhob[j]), weights=self.phi[i,j], mode='nearest')*self.delta
            self.Fec = np.sum(aux)*self.delta

        self.F = self.Fid+self.Fhs+self.Fec+self.Fcoul

    def Calculate_Omega(self):
        self.Calculate_Free_energy()
        self.Omega = (self.F + np.sum((self.Vext-self.mu[:,np.newaxis])*self.rho)*self.delta)/self.L

    def Calculate_Potential(self):
        self.q = np.sum(self.rho*self.Z[:,np.newaxis],axis=0)
        # self.psi[:] = self.poisson.ElectrostaticPotential(self.q,np.array([self.psi0,0.0]))
        # self.sigma = -np.sum(self.q)*self.delta
        if self.psi0 != 0.0: 
            Q = np.cumsum(self.q[::-1])
            self.psi[:] = self.psi0 + (4*np.pi*self.lB)*self.x*Q[::-1]*self.delta +(4*np.pi*self.lB)*np.cumsum(self.x*self.q)*self.delta
            self.sigma = -np.sum(self.q)*self.delta
        else:
            Q = np.cumsum(self.q[::-1])
            self.psi[:] = (4*np.pi*self.lB)*self.x*Q[::-1]*self.delta +(4*np.pi*self.lB)*np.cumsum(self.x*self.q)*self.delta
            # self.sigma = -np.sum(self.q)*self.delta
            psi0new = self.psi[1] + (4*np.pi*self.lB)*self.sigma[0]*self.delta
            self.psi[:] = self.psi + psi0new
            # Q = np.cumsum(self.x[::-1]*self.q[::-1])
            # self.psi[:] = -(4*np.pi*self.lB)*self.sigma[0]*self.x - (4*np.pi*self.lB)*Q[::-1]*self.delta -(4*np.pi*self.lB)*self.x*np.cumsum(self.q)*self.delta

    def Calculate_c1(self):
        # HS c1
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

        for i in range(self.species):
            if self.ecmethod == 'PB':
                self.c1ec[i,:] = 0.0
            elif self.ecmethod == 'BFD':
                self.c1ec[i,:] = 0.0
                for j in range(self.species):
                    self.c1ec[i,:] += -convolve1d((self.rho[j,:]-self.rhob[j]), weights=self.phi[i,j], mode='nearest')*self.delta
            elif self.ecmethod == 'fMSA' or self.ecmethod == 'fMSA-asymmetrical':
                dPhieledn = -self.lB*(self.Z[i]**2*self.Gamma+2*self.d[i]*self.Eta*self.Z[i]-self.Eta**2*self.d[i]**3*(2.0/3.0-self.Gamma*self.d[i]/3.0))/(1+self.Gamma*self.d[i])
                self.c1ec[i,:] = -convolve1d(dPhieledn, weights=self.ws[i], mode='nearest')*self.delta
                del dPhieledn
                for j in range(self.species):
                    self.c1ec[i,:] += -convolve1d((self.rho[j,:]-self.rhob[j]), weights=self.phi[i,j], mode='nearest')*self.delta
        
        # Coulomb c1
        self.c1coul = -self.Z[:,np.newaxis]*self.psi

        self.c1exc = self.c1hs+self.c1ec
        self.c1 = self.c1exc+self.c1coul

    def Calculate_mu(self):
        # HS chemical potential
        n3 = np.sum(self.rhob*np.pi*self.d**3/6)
        n2 = np.sum(self.rhob*np.pi*self.d**2)
        n1 = np.sum(self.rhob*self.d/2)
        n0 = np.sum(self.rhob)

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

        dPhidn0 = -np.log(1-n3)
        dPhidn1 = n2*phi2/(1-n3)
        dPhidn2 = n1*phi2/(1-n3) + (3*n2**2)*phi3/(24*np.pi*(1-n3)**2)
        dPhidn3 = n0/(1-n3) +(n1*n2)*(dphi2dn3 + phi2/(1-n3))/(1-n3) + (n2**3)*(dphi3dn3+2*phi3/(1-n3))/(24*np.pi*(1-n3)**2)

        self.muid = np.log(self.rhob)

        self.muhs = dPhidn0+dPhidn1*self.d/2+dPhidn2*np.pi*self.d**2+dPhidn3*np.pi*self.d**3/6

        # EC chemical potential
        if self.ecmethod == 'PB' or self.ecmethod == 'BFD':
            self.muec = np.zeros_like(self.Z)
        else:
            self.muec = -self.lB*(self.Z**2*self.Gammabulk+2*self.d*self.Etabulk*self.Z-self.Etabulk**2*self.d**3*(2.0/3.0-self.Gammabulk*self.d/3.0))/(1+self.Gammabulk*self.d)

        self.muexc = self.muhs+self.muec
        self.mu = self.muid + self.muexc

    def Calculate_Equilibrium(self,method='picard',logoutput=False):
        return Optimize(self,method=method,logoutput=logoutput)



if __name__ == "__main__":
    test1 = False # the MSA screening parameter 
    test2 = True # Voukadinova data

    import matplotlib.pyplot as plt
    import timeit

    starttime = timeit.default_timer()

    if test1: 
        c = np.linspace(1e-3,1.0,1000)
        Z = np.array([-1,1])
        a = np.array([0.3,0.3])
        lB = 0.714

        gamma0 = np.zeros_like(c)
        gamma1 = np.zeros_like(c)
        rhoarray = np.zeros((2,c.size))
        for i in range(c.size):
            rho = np.array([-(Z[1]/Z[0])*c[i],c[i]])*0.622
            rhoarray[:,i] = rho
            kappa = np.sqrt(4*np.pi*lB*np.sum(Z**2*rho))
            # amed = np.power(np.sum(rho*a**3)/np.sum(rho),1.0/3.0)
            amed = np.sqrt(np.sum(rho*a**2)/np.sum(rho))
            gamma0[i] = (np.sqrt(1+2*kappa*amed)-1)/(2*amed)
        gamma1 = Gammaparameter(rhoarray,a,Z,lB)

        # plt.plot(c,gamma1,'k')
        # plt.plot(c,gamma0,'--',color='grey')
        # plt.xlabel('c (mol/L)')
        # plt.ylabel('$\Gamma_b$ (nm$^{-1}$)')
        # plt.show()

    ##################################################################################
    if test2: 
        d = np.array([0.3,0.3])
        delta = 0.01*d[1]
        L = 6.5
        N = int(L/delta)
        Z = np.array([-1,3])

        c = 0.01 #mol/L (equivalent to ionic strength for 1:1)
        rhob = np.array([-(Z[1]/Z[0])*c,c])*6.022e23/1.0e24 # particles/nm^3

        ele = ElectrolyteDFT(N,delta,d=d,Z=Z,rhob=rhob,ecmethod='BFD')

        # sigma = -0.1704/d[0]**2
        sigma = -3.125 

        ele.Set_Boundary_Conditions(sigma,0.0)
        ele.Set_InitialCondition()
        ele.Update_System()

        plt.plot(ele.x,ele.rho[0]/rhob[0])
        plt.plot(ele.x,ele.rho[1]/rhob[1],'C3')
        plt.show()

        plt.plot(ele.x,ele.psi)
        plt.show()

        ele.Calculate_Equilibrium(method='anderson',logoutput=True)

        np.save('results/profiles-BFD-Voukadinova2018-electrolyte-Fig5-Z+=3-rho+=0.01M.npy',[ele.x,ele.rho[0],ele.rho[1],ele.psi,ele.c1exc[0]+ele.muexc[0],ele.c1exc[1]+ele.muexc[1]])
    
    print("time :", timeit.default_timer() - starttime, 'sec')