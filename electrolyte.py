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

" Global variables for the FIRE algorithm"
Ndelay = 20
Nmax = 10000
finc = 1.1
fdec = 0.5
fa = 0.99
Nnegmax = 2000

@vectorize([float32(float32),float64(float64)])
def phi2func(eta):
    if eta <= 1e-3: return 1+eta**2/9
    else: return 1+(2*eta-eta**2+2*np.log(1-eta)*(1-eta))/(3*eta)

@vectorize([float32(float32),float64(float64)])
def phi3func(eta):
    if eta <= 1e-3: return 1-4*eta/9
    else: return 1-(2*eta-3*eta**2+2*eta**3+2*np.log(1-eta)*(1-eta)**2)/(3*eta**2)

@vectorize([float32(float32),float64(float64)])
def phi1func(eta):
    if eta <= 1e-3: return 1-2*eta/9-eta**2/18
    else: return 2*(eta+np.log(1-eta)*(1-eta)**2)/(3*eta**2)

@vectorize([float32(float32),float64(float64)])
def dphi1dnfunc(eta):
    if eta <= 1e-3: return -2/9-eta/9-eta**2/15.0
    else: return (2*(eta-2)*eta+4*(eta-1)*np.log(1-eta))/(3*eta**3)

@vectorize([float32(float32),float64(float64)])
def dphi2dnfunc(eta):
    if eta <= 1e-3: return 2*eta/9+eta**2/6.0
    else: return -(2*eta+eta**2+2*np.log(1-eta))/(3*eta**2)

@vectorize([float32(float32),float64(float64)])
def dphi3dnfunc(eta):
    if eta <= 1e-3: return -4.0/9+eta/9
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
    def __init__(self,N,delta,lB=0.714,d=np.array([1.0,1.0]),Z=np.array([-1,1]),rhob=np.array([0.1,0.1]),fmtmethod='WBI',ecmethod='fMSA-symmetrical'):
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
        self.q0 = np.zeros((self.species,self.N),dtype=np.float32)
        self.phi = np.zeros((self.species,self.species),dtype=object)
        self.phiint = np.zeros((self.species,self.species),dtype=np.float32)
        self.c1long = np.zeros((self.species,self.N),dtype=np.float32)
        self.c1ec = np.zeros((self.species,self.N),dtype=np.float32)
        self.ws = np.zeros(self.species,dtype=object)
        
        self.Gamma = np.zeros(N,dtype=np.float32)
        self.Eta = np.zeros(N,dtype=np.float32)
        self.Psi = np.zeros(N,dtype=np.float32)

        self.poisson = Poisson1D(self.N,self.delta,boundary_condition='mixed')
        self.Gammabulk = Gammabulkparameter(self.rhob,self.d,self.Z,self.lB)
        self.Etabulk = Etafunc(self.rhob,self.d,self.Z,self.Gammabulk)
        print('inverse Gammabulk = ',1.0/self.Gammabulk,' nm')

        # auxiliary variables for linear system of electrostatic potential
        self.Ab = np.zeros((3,self.N))
        self.r = np.zeros(self.N)
        self.Ab[0,1:] = 1.0
        self.Ab[2,:-1] = 1.0
        # self.Ab[0,1] = -1.0
        self.Ab[2,-2] = 0.0
        self.Ab[1,:] = -2.0
        self.Ab[1,0] = -1.0
        self.Ab[1,-1] = -1.0

        if self.ecmethod == 'fMSA-symmetrical':
            self.b = self.d+1.0/self.Gammabulk
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

            self.b = self.d+1.0/self.Gammabulk

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

    def Update_System(self):
        self.Calculate_auxiliary_quantities()
        self.Calculate_c1()
        self.Calculate_Omega()

    def Set_External_Potential(self,Vext=0.0):
        self.Vext[:] = Vext

    def Set_Boundary_Conditions(self,sigma=0.0,psibulk=0.0):
        self.bound_value = np.array([sigma,psibulk])

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

        if self.ecmethod != 'BFD':
            for i in range(self.species):
                self.q0[i,:] = convolve1d(self.rho[i], weights=self.ws[i], mode='nearest')*self.delta
            self.Gamma = Gammaparameter(self.q0,self.d,self.Z,self.lB) 
            self.Eta = Etafunc(self.q0,self.d,self.Z,self.Gamma)
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

        aux = convolve1d(self.Psi, weights=[-1,1], mode='nearest')/self.delta
        self.Fcoul = -(1/(8*np.pi*self.lB))*np.sum(aux**2)*self.delta

        aux = np.sum(self.rho[:,:]*self.Z[:,np.newaxis],axis=0)
        self.Fint = np.sum(aux*self.Psi)*self.delta 

        if self.ecmethod == 'fMSA-symmetrical' or self.ecmethod == 'fMSA-asymmetrical':
            aux = self.Gamma**3/(3*np.pi)
            for i in range(self.species):
                aux += -self.lB*self.q0[i,:]*self.Z[i]*(self.Z[i]*self.Gamma+self.Eta*self.d[i])/(1+self.Gamma*self.d[i])
                for j in range(self.species):
                    aux += 0.5*(self.rho[i,:]-self.rhob[i])*convolve1d((self.rho[j,:]-self.rhob[j]), weights=self.phi[i,j], mode='nearest')*self.delta
        elif self.ecmethod == 'BFD':
            aux = np.ones_like(self.x)*(self.Gammabulk**3/(3*np.pi) - np.sum(self.lB*self.rhob*self.Z*(self.Z*self.Gammabulk+self.Etabulk*self.d)/(1+self.Gammabulk*self.d)))
            for i in range(self.species):
                for j in range(self.species):
                    aux += 0.5*(self.rho[i,:]-self.rhob[i])*convolve1d((self.rho[j,:]-self.rhob[j]), weights=self.phi[i,j], mode='nearest')*self.delta
        self.Fec = np.sum(aux)*self.delta

        self.F = self.Fid+self.Fhs+self.Fcoul+self.Fint+self.Fec

    def Calculate_Omega(self):
        self.Calculate_Free_energy()
        self.Omega = (self.F + np.sum((self.Vext-self.mu[:,np.newaxis])*self.rho)*self.delta)/self.L

    def Calculate_Potential(self):
        # solving the potential
        self.r[:] = -4*np.pi*self.lB*self.delta**2*np.sum(self.rho*self.Z[:,np.newaxis],axis=0)
        self.r[0] += -4*np.pi*self.lB*self.delta*self.bound_value[0] # surface charge specified by the user
        self.r[-1] = -self.bound_value[1] # surface potential specified by the user

        self.Psi[:] = solve_banded((1,1), self.Ab, self.r)

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
            if self.ecmethod == 'BFD':
                self.c1ec[i,:] = 0.0
            else:
                dPhieledn = -self.lB*(self.Z[i]**2*self.Gamma+2*self.d[i]*self.Eta*self.Z[i]-self.Eta**2*self.d[i]**3*(2.0/3.0-self.Gamma*self.d[i]/3.0))/(1+self.Gamma*self.d[i])
                self.c1ec[i,:] = -convolve1d(dPhieledn, weights=self.ws[i], mode='nearest')*self.delta
                del dPhieledn

            for j in range(self.species):
                self.c1ec[i,:] += -convolve1d((self.rho[j,:]-self.rhob[j]), weights=self.phi[i,j], mode='nearest')*self.delta
        
        # Coulomb c1
        self.c1coul = -self.Z[:,np.newaxis]*self.Psi

        self.c1exc = self.c1hs+self.c1ec
        self.c1 = self.c1hs+self.c1ec+self.c1coul

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
        if self.ecmethod == 'BFD':
            self.muec = 0.0*self.Z
        else:
            self.muec = -self.lB*(self.Z**2*self.Gammabulk+2*self.d*self.Etabulk*self.Z-self.Etabulk**2*self.d**3*(2.0/3.0-self.Gammabulk*self.d/3.0))/(1+self.Gammabulk*self.d)

        self.muexc = self.muhs+self.muec
        self.mu = self.muid + self.muexc

    def Calculate_Equlibrium(self,method='picard',alpha0=0.4,atol=1e-6,dt=0.1,logoutput=False):

        self.Update_System()

        if method == 'fire':
            error = 10*atol 
            dtmax = 10*dt
            dtmin = 0.02*dt
            alpha = alpha0
            Npos = 0
            Nneg = 0

            lnrho = np.log(self.rho)
            V = np.zeros((self.species,self.N),dtype=np.float32)
            self.Update_System()
            F = -self.rho*(lnrho -self.c1 - self.mu[:,np.newaxis]+self.Vext)*self.delta/self.L

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
                    lnrho = lnrho - 0.5*dt*V
                    V = np.zeros((self.species,self.N),dtype=np.float32)
                    self.rho[:] = np.exp(lnrho)
                    self.Update_System()

                V = V + 0.5*dt*F
                V = (1-alpha)*V + alpha*F*np.linalg.norm(V)/np.linalg.norm(F)
                lnrho = lnrho + dt*V
                self.rho[:] = np.exp(lnrho)
                self.Update_System()
                F = -self.rho*(lnrho -self.c1 - self.mu[:,np.newaxis]+self.Vext)*self.delta/self.L
                V = V + 0.5*dt*F

                error = max(np.abs(F.min()),F.max())
                if error < atol: break

                if logoutput: print(i,self.Omega,error)

            del V, F  

        elif method == 'picard':
            alpha = alpha0
            lnrho = np.log(self.rho) 

            nsig = np.array([int(0.5*self.d[0]/self.delta),int(0.5*self.d[1]/self.delta)])      

            for i in range(Nmax):
                F = -(lnrho -self.c1 - self.mu[:,np.newaxis]+self.Vext)
                F[0,:nsig[0]] = 0.0
                F[1,:nsig[1]] = 0.0

                lnrho[:] = lnrho + alpha*F
                self.rho[:] = np.exp(lnrho)
                self.Update_System()

                error = max(abs(F.min()),F.max())
                alpha = min(0.02,alpha0/(atol/error)**0.5)
                if error < atol: break
                if logoutput: print(i,self.Omega,error)

        elif method == 'anderson':
            m = 5
            beta0 = (1.0/m)*np.ones(m)
            beta = beta0.copy()

            x = np.log(self.rho)
            x0 = x.copy()

            xstr = [np.zeros_like(x)]*m
            ustr = [np.zeros_like(x)]*m
            Fstr = [np.zeros_like(x)]*m

            nsig = np.array([int(0.5*self.d[0]/self.delta),int(0.5*self.d[1]/self.delta)])  

            for i in range(Nmax):

                F = -(x -self.c1 - self.mu[:,np.newaxis]+self.Vext)
                F[0,:nsig[0]] = 0.0
                F[1,:nsig[1]] = 0.0
                u = self.c1+ self.mu[:,np.newaxis]- self.Vext
                u[0,:nsig[0]] = 0.0
                u[1,:nsig[1]] = 0.0
                
                if i < m:
                    xstr[i] = x
                    Fstr[i] = F
                    ustr[i] = u
                else:
                    xstr[:m-1] = xstr[1:m]
                    xstr[m-1] = x
                    Fstr[:m-1] = Fstr[1:m]
                    Fstr[m-1] = F
                    ustr[:m-1] = ustr[1:m]
                    ustr[m-1] = u

                    def objective(alp):
                        fobj = 0.0
                        for l in range(m):
                            fobj += (alp[l]**2)*np.linalg.norm(Fstr[l])
                        return fobj

                    res = optimize.minimize(objective, np.sqrt(beta0), method='Nelder-Mead', tol=1e-2)
                    beta[:] = res.x**2/np.sum(res.x**2)
                    print(beta)

                    x[:] = (1-alpha0)*beta[0]*xstr[0] + alpha0*beta[0]*ustr[0]
                    for l in range(1,m):
                        x += (1-alpha0)*beta[l]*xstr[l] + alpha0*beta[l]*ustr[l]

                    self.rho[:] = np.exp(x)
                    plt.plot(self.x,self.Psi)
                    plt.show()


                    plt.plot(self.x,u[0,:]/self.rhob[0])
                    plt.plot(self.x,u[1,:]/self.rhob[1])
                    plt.show()
                    self.Update_System()
                    self.Calculate_Potential()

                error = max(np.abs(F.min()),F.max())
                if logoutput: print(i,self.Omega,error)
                if error < atol: break

            del F, xstr, Fstr, ustr

        return i



if __name__ == "__main__":
    test0 = False # the MSA screening parameter 
    test1 = False # Voukadinova
    test2 = True # using Poisson1D

    import matplotlib.pyplot as plt
    from fire import optimize_fire2
    from andersonacc import optimize_anderson
    from fmt1d import FMTplanar
    from pb import PBplanar
    from poisson1d import Poisson1D
    import timeit

    starttime = timeit.default_timer()

    if test0: 
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
        
    if test1: 
        d = np.array([0.3,0.15])
        delta = 0.025*d[1]
        L = 4.0 + max(d)
        N = int(L/delta)
        Z = np.array([-1,3])

        c = 0.1 #mol/L (equivalent to ionic strength for 1:1)
        rhob = np.array([-(Z[1]/Z[0])*c,c])*6.022e23/1.0e24 # particles/nm^3

        x = np.linspace(0,L,N)

        # sigma = -0.1704/d[0]**2
        sigma = -3.125

        n = np.ones((2,N),dtype=np.float32)
        nsig = np.array([int(0.5*d[0]/delta),int(0.5*d[1]/delta)])

        param = np.array([rhob[0],rhob[1],sigma])

        # Here we will solve the PB equation as a input to DFT
        pb = PBplanar(N,delta,species=2,d=d,Z=Z)
        kD = np.sqrt(4*np.pi*pb.lB*np.sum(Z**2*rhob))

        def Fpsi(psi,param):
            n[0,:nsig[0]] = 1.0e-16
            n[1,:nsig[1]] = 1.0e-16
            n[0,nsig[0]:] = param[0]*np.exp(-Z[0]*psi[nsig[0]:])
            n[1,nsig[1]:] = param[1]*np.exp(-Z[1]*psi[nsig[1]:])
            sigma = param[2]
            Fele = pb.free_energy(n,psi)
            return (Fele+sigma*psi[0])/L

        def dFpsidpsi(psi,param):
            n[0,:nsig[0]] = 1.0e-16
            n[1,:nsig[1]] = 1.0e-16
            n[0,nsig[0]:] = param[0]*np.exp(-Z[0]*psi[nsig[0]:])
            n[1,nsig[1]:] = param[1]*np.exp(-Z[1]*psi[nsig[1]:])
            sigma = param[2]
            return -pb.dOmegadpsi(n,psi,[sigma,0.0])*delta/L

        psi0 = 0.1*sigma*4*np.pi*pb.lB # attenuation of the surface charge
        psi = np.zeros(N,dtype=np.float32)
        psi[:nsig[0]] = psi0*(1/kD+0.5*d[0]-x[:nsig[0]])
        psi[nsig[0]:] = psi0*np.exp(-kD*(x[nsig[0]:]-0.5*d[0]))/kD
    
        [varsol,Omegasol,Niter] = optimize_fire2(psi,Fpsi,dFpsidpsi,param,atol=1.0e-8,dt=0.02,logoutput=False)
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

        fmt = FMTplanar(N,delta,species=2,d=d)
        ele = ElectrolyteDFT(N,delta,species=2,d=d,Z=Z,rhob=rhob,model='symmetrical')

        # solving the electrostatic potential equation
        def Fpsi2(psi,nn):
            Fele = ele.Flong(psi)+ele.Fint(nn,psi)
            return (Fele-sigma*psi[0])/L

        def dFpsidpsi2(psi,nn):
            return -ele.dOmegadpsi(nn,psi,[sigma,0.0])*delta/L

        mu = np.log(rhob) + fmt.mu(rhob) + ele.mu(rhob)

        # Now we will solve the DFT equations
        def Omega(var,psi):
            nn[0,:] = np.exp(var[0])
            nn[1,:] = np.exp(var[1])
            Fid = np.sum(nn*(var-1.0))*delta
            Fhs = np.sum(fmt.Phi(nn))*delta
            Fele = ele.free_energy(nn,psi)
            return (Fid+Fhs+Fele-np.sum(mu[:,np.newaxis]*nn*delta)-sigma*psi[0])/L

        def dOmegadnR(var,psi):
            nn[0,:] = np.exp(var[0])
            nn[1,:] = np.exp(var[1])

            [varsol2,Omegasol2,Niter] = optimize_fire2(psi,Fpsi2,dFpsidpsi2,nn,atol=1.0e-8,dt=0.005,logoutput=False)
            psi[:] = varsol2-varsol2[-1]

            c1hs = fmt.c1(nn)
            c1ele = ele.c1(nn,psi)
            return nn*(var -c1hs -c1ele - mu[:,np.newaxis])*delta/L

        muMSA = ele.muMSA(rhob)

        var = np.log(n)
        [varsol,Omegasol1,Niter] = optimize_fire2(var,Omega,dOmegadnR,psi,alpha0=alpha,atol=1.0e-5,dt=0.02,logoutput=True)
        n[0,:] = np.exp(varsol[0])
        n[1,:] = np.exp(varsol[1])

        c1MSA = ele.c1MSA(n)+muMSA[:,np.newaxis]

        np.save('DFTresults/profiles-fMSA-Voukadinova2018-electrolyte-Fig3-Z+=3-rho+=0.1M.npy',[x,n[0],n[1],psi,c1MSA[0],c1MSA[1]])

    ##################################################################################
    if test2: 
        d = np.array([0.3,0.3])
        delta = 0.01*d[1]
        L = 12.5
        N = int(L/delta)
        Z = np.array([-1,1])

        c = 0.01 #mol/L (equivalent to ionic strength for 1:1)
        rhob = np.array([-(Z[1]/Z[0])*c,c])*6.022e23/1.0e24 # particles/nm^3

        x = np.linspace(0,L,N)

        ele = ElectrolyteDFT(N,delta,d=d,Z=Z,rhob=rhob,ecmethod='BFD')

        # sigma = -0.1704/d[0]**2
        sigma = -3.125

        nsig = np.array([int(0.5*d[0]/delta),int(0.5*d[1]/delta)])      

        kD = np.sqrt(4*np.pi*ele.lB*np.sum(Z**2*rhob))
        psi0 = 0.1*sigma*4*np.pi*ele.lB/kD # attenuation of the surface charge
        ele.Psi = psi0*np.exp(-kD*x)

        ele.Set_Boundary_Conditions(sigma,ele.Psi[-1])

        # plt.plot(x,ele.Psi)
        # plt.show()

        ele.rho[0,:nsig[0]] = 1.0e-16
        ele.rho[1,:nsig[1]] = 1.0e-16
        ele.rho[0,nsig[0]:] = rhob[0]*np.exp(-Z[0]*ele.Psi[nsig[0]:])
        ele.rho[1,nsig[1]:] = rhob[1]*np.exp(-Z[1]*ele.Psi[nsig[1]:])
        # n[0,nsig[0]:] = param[0]
        # n[1,nsig[1]:] = param[1]
        ele.Update_System()

        # plt.plot(x,ele.rho[0]/rhob[0])
        # plt.plot(x,ele.rho[1]/rhob[1],'C3')
        # plt.show()

        # plt.plot(x,ele.Psi)
        # plt.show()

        ele.Calculate_Equlibrium(method='picard',alpha0=0.2,atol=1e-4,dt=0.02,logoutput=True)

        np.save('DFTresults/profiles-BFD-Voukadinova2018-electrolyte-Fig5-Z+=1-rho+=0.01M.npy',[x,ele.rho[0],ele.rho[1],ele.Psi,ele.c1exc[0]+ele.muexc[0],ele.c1exc[1]+ele.muexc[1]])
    
    print("time :", timeit.default_timer() - starttime)