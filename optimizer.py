#!/usr/bin/env python3

# This script is the python implementation of the Density Functional Theory
# for Electrolyte Solution in the presence of an external electrostatic potential
#
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2021-06-02
# Updated: 2022-08-03
# Version: 2.0
#
import numpy as np

" Global variables for the FIRE algorithm"
Ndelay = 20
Nmax = 10000
finc = 1.1
fdec = 0.5
fa = 0.99
Nnegmax = 2000


def Optimize(dft,method='picard',logoutput=False):
    dft.Update_System()

    if method == 'picard':
        atol=1.e-4
        rtol = 1.e-3
        alpha = 0.01
        lnrho = np.log(dft.rho) 

        nsig = (0.5*dft.d/dft.delta).astype(int)  
        errorlast = np.inf

        for i in range(Nmax):
            lnrhonew = dft.c1 + dft.mu[:,np.newaxis] - dft.Vext
            lnrhonew[0,:nsig[0]] = np.log(1.0e-16)
            lnrhonew[1,:nsig[1]] = np.log(1.0e-16)
            
            lnrho[:] = (1-alpha)*lnrho + alpha*lnrhonew
            dft.rho[:] = np.exp(lnrho)
            dft.Update_System()

            F = (lnrho - lnrhonew)
            error = np.linalg.norm(F/(atol+rtol*np.abs(lnrho)))

            if errorlast > error:  alpha = min(0.02,alpha*finc)
            else: alpha = max(1.0e-3,alpha*fdec)

            errorlast = error
            if error < 1.0: break
            if logoutput: print(i,dft.Omega,error)

    elif method == 'fire':
        if dft.ecmethod == 'PB': alpha0 = 0.2
        else: alpha0 = 0.25
        rtol = 1.e-6
        atol = 1.e-5
        dt = 0.45
        dtmax = 2*dt
        dtmin = 0.01*dt
        alpha = alpha0
        Npos = 0
        Nneg = 0

        lnrho = np.log(dft.rho)
        V = np.zeros_like(dft.rho)
        F = -dft.rho*(lnrho -dft.c1 -dft.mu[:,np.newaxis]+dft.Vext)*dft.delta/dft.L

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
                lnrho[:] = lnrho - 0.5*dt*V
                V = np.zeros((dft.species,dft.N),dtype=np.float32)
                dft.rho[:] = np.exp(lnrho)
                dft.Update_System()

            V[:] = V + 0.5*dt*F
            V[:] = (1-alpha)*V + alpha*F*np.linalg.norm(V)/np.linalg.norm(F)
            lnrho[:] = lnrho + dt*V
            error = np.linalg.norm(F/(atol+rtol*np.abs(lnrho)))

            dft.rho[:] = np.exp(lnrho)
            dft.Update_System()
            F[:] = -dft.rho*(lnrho -dft.c1 - dft.mu[:,np.newaxis]+dft.Vext)*dft.delta/dft.L
            V[:] = V + 0.5*dt*F

            if error < 1.0: break

            if logoutput: print(i,dft.Omega,error)

        del V, F  

    return i