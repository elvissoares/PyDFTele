#!/usr/bin/env python3

# This script is the python implementation of the Poisson-Boltzmann equation
#
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2021-06-02
# Updated: 2021-12-13
# Version: 1.0
# https://www.scirp.org/pdf/JEMAA_2014092510103331.pdf
import sys
import numpy as np
from scipy.linalg import solve_banded

" The PB model for electrolyte solutions using the generalized grand potential"

class Poisson1D():
    def __init__(self,N,delta,lB=0.714,boundary_condition='dirichlet'):
        self.N = N
        self.delta = delta
        self.L = delta*N
        self.boundary_condition = boundary_condition

        self.lB = lB # in nm (for water)

        # auxiliary variables for linear system
        self.Ab = np.zeros((3,self.N))
        # self.r = np.zeros(self.N)

    def ElectrostaticPotential(self,q,bound_value):

        self.r = -4*np.pi*self.lB*self.delta**2*q  

        if self.boundary_condition == 'dirichlet':
            # surface potential specified by the user
            self.r[0] = bound_value[0]
            # self.r[-1] = -bound_value[1]

            self.Ab[0,1:] = 1.0
            self.Ab[2,:-1] = 1.0
            self.Ab[0,1] = 0.0
            self.Ab[1,:] = -2.0
            self.Ab[1,0] = 1.0

            psi = solve_banded((1,1), self.Ab, self.r)
            # psi[:] = tridiag(self.Ab[2,:],self.Ab[1,:],self.Ab[0,:], self.r)

        elif self.boundary_condition == 'neumann':
            # surface charge specified by the user
            self.r[0] += -4*np.pi*self.lB*self.delta*bound_value[0]
            self.r[-1] += -4*np.pi*self.lB*self.delta*bound_value[-1]

            self.Ab[0,1:] = 1.0
            self.Ab[2,:-1] = 1.0
            self.Ab[0,1] = -1.0
            self.Ab[2,-2] = -1.0
            self.Ab[1,:] = -2.0
            self.Ab[1,0] = 1.0
            self.Ab[1,-1] = 1.0

            psi = solve_banded((1,1), self.Ab, self.r)

        elif self.boundary_condition == 'mixed':
            self.r[0] = -0.5*4*np.pi*self.lB*self.delta**2*q[0]
            self.r[0] += -4*np.pi*self.lB*self.delta*bound_value[0] # surface charge specified by the user
            self.r[-1] += -bound_value[1] # surface potential specified by the user

            self.Ab[0,1:] = 1.0
            self.Ab[2,:-1] = 1.0
            self.Ab[1,:] = -2.0
            self.Ab[1,0] = -1.0

            psi = solve_banded((1,1), self.Ab, self.r)

        else:
            raise ValueError('Boundary condition for psi not recognized')   
        
        return psi


if __name__ == "__main__":
    test1 = False # Dirichlet boundary condition
    test2 = True # Neumann boundary condition
    test3 = False # Mixed boundary condition

    import matplotlib.pyplot as plt
    from fire import optimize_fire2

    if test1: 
        delta = 0.01
        L = 2*np.pi
        N = int(L/delta)

        pois = Poisson1D(N,delta,boundary_condition='dirichlet')
        x = np.linspace(0,L,N)

        q = np.empty(N,dtype=np.float32)
        q[:] = (1.0/(4*np.pi*pois.lB))*(2*np.sin(x)+x*np.cos(x))

        psi_exact = x*np.cos(x)

        psi0 = np.array([psi_exact[0],psi_exact[-1]])

        # plt.plot(x,q)
        # plt.show()

        psi = np.zeros(N,dtype=np.float32)
        # psi[:] = psi0*np.exp(-kD*x)

        psi[:] = pois.ElectrostaticPotential(q,psi0)

        plt.plot(x,psi,'k',label='numerical')
        plt.scatter(x[::5],psi_exact[::5],edgecolors='C0',facecolor='None',label='exact')
        plt.ylim(-4,8)
        plt.xlim(0,2*np.pi)
        plt.ylabel(r'$\psi(x)$')
        plt.xlabel(r'$x$')
        plt.legend(loc='upper left')
        plt.show()


        k = 4/np.pi
        q[:] = (1.0/(4*np.pi*pois.lB))*(k*(2-k*x)*np.exp(-k*x))

        psi_exact = x*np.exp(-k*x)

        psi0 = np.array([psi_exact[0],psi_exact[-1]])

        psi[:] = pois.ElectrostaticPotential(q,psi0)

        plt.plot(x,psi,'k',label='numerical')
        plt.scatter(x[::5],psi_exact[::5],edgecolors='C3',facecolor='None',label='exact')
        # plt.ylim(-4,8)
        plt.xlim(0,2*np.pi)
        plt.ylabel(r'$\psi(x)$')
        plt.xlabel(r'$x$')
        plt.legend(loc='best')
        plt.show()

    if test2: 
        delta = 0.0001
        L = 2*np.pi
        N = int(L/delta)

        pois = Poisson1D(N,delta,boundary_condition='mixed')
        x = np.linspace(0,L,N)

        q = np.empty(N,dtype=np.float32)
        q[:] = (1.0/(4*np.pi*pois.lB))*(2*np.sin(x)+x*np.cos(x))

        psi_exact = x*np.cos(x)

        sigma = np.array([(1.0/(4*np.pi*pois.lB))*(psi_exact[0]-psi_exact[1])/delta,psi_exact[-1]])

        # plt.plot(x,q)
        # plt.show()

        psi = np.zeros(N,dtype=np.float32)
        # psi[:] = psi0*np.exp(-kD*x)

        psi[:] = pois.ElectrostaticPotential(q,sigma)

        plt.plot(x,psi,'k',label='numerical')
        plt.scatter(x[::5],psi_exact[::5],edgecolors='C0',facecolor='None',label='exact')
        plt.ylim(-4,8)
        plt.xlim(0,2*np.pi)
        plt.ylabel(r'$\psi(x)$')
        plt.xlabel(r'$x$')
        plt.legend(loc='upper left')
        plt.show()

        k = 4/np.pi
        q[:] = (1.0/(4*np.pi*pois.lB))*(k*(2-k*x)*np.exp(-k*x))

        psi_exact = x*np.exp(-k*x)

        sigma = np.array([(1.0/(4*np.pi*pois.lB))*(psi_exact[0]-psi_exact[1])/delta,psi_exact[-1]])

        psi[:] = pois.ElectrostaticPotential(q,sigma)

        plt.plot(x,psi,'k',label='numerical')
        plt.scatter(x[::5],psi_exact[::5],edgecolors='C3',facecolor='None',label='exact')
        # plt.ylim(-4,8)
        plt.xlim(0,2*np.pi)
        plt.ylabel(r'$\psi(x)$')
        plt.xlabel(r'$x$')
        plt.legend(loc='best')
        plt.show()