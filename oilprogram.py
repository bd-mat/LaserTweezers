# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 18:09:13 2025

@author: bjama
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

T = 296 # Kelvin
kB = 1.381e-23 # J/K
c = 0.091e-6 # m/pixel
dc = 0.001e-6 # c uncertainty

# error on a single measurement is c/2
derr = c/2

# FPS
FPS = 310.1

if FPS==310.1:
    std = 0.6425
else:
    std = 0.485

derr += c*std

# viscosity values
vx = 0.5647e-3
vx_er = 0.2143e-3
vy = 0.792e-3
vy_er = 0.4392e-3
# calculate cutoff
cutoff = 1 # s
N = int(cutoff*np.ceil(FPS))

FILE1 = 'brown'
FILE2 = '_vals.csv'

def read_data(filename):
    data = np.genfromtxt(filename,dtype=float,delimiter=',',skip_header=True)
    return data

def MSD(data,derr):
    MSD = np.zeros(len(data)-1)
    unc = np.zeros_like(MSD)
    N = len(data)
    for n in range(0,N-1):
        for i in range(n,N-n):
            temp = (data[i+n]-data[i])**2
            MSD[n] += temp
            unc[n] += (2*np.sqrt(2)*derr)*np.sqrt(temp)
        MSD[n] = MSD[n]/(N-n)
        unc[n] = unc[n]/(N-n)
    return MSD,unc

def examplefunc(x,a):
    return a*x

def main():
    data = read_data(FILE1+FILE2)
    tvals = data[:,0]
    xvals = data[:,1]
    yvals = data[:,2]
    # convert to SI
    tvals = tvals*(1/FPS)
    xvals = xvals*c
    yvals = yvals*c
    # convert to displ.
    xmean = np.mean(xvals)
    ymean = np.mean(yvals)
    xvals = xvals - xmean
    yvals = yvals - ymean
    # check for brownian
    check = input("Is this brownian data?")
    if check == 'y':
        check = input("Have you already run MSD?")
        if check == 'y':
            print("Assuming naming convention.")
            MSDr = read_data(FILE1+"_MSD.csv")
            tau = MSDr[:,0]
            MSDx = MSDr[:,1]
            unc_x = MSDr[:,2]
            MSDy = MSDr[:,3]
            unc_y = MSDr[:,4]
        else:
            # finds msds
            MSD_resx = MSD(xvals,derr)
            MSD_resy = MSD(yvals,derr)
            MSDx = MSD_resx[0]
            MSDy = MSD_resy[0]
            unc_x = MSD_resx[1]
            unc_y = MSD_resy[1]
            tau = (1/FPS)*np.linspace(0,len(MSDx)-1,num=len(MSDx))
            check = input("Save data?")
            if check == 'y':
                file = open(FILE1+'_MSD.csv','w')
                file.write('tau,MSDx,MSDx_error,MSDy,MSDy_error\n')
                for i in range(0,len(MSDx)):
                    file.write(str(tau[i])+','+str(MSDx[i])+"," + str(unc_x[i]) + ","
                               +str(MSDy[i])+ "," + str(unc_y[i]) +"\n")
                file.close()
        # plot msds (no cutoff)
        plt.errorbar(tau,MSDx,unc_x)
        plt.title('X MSD')
        plt.xlabel(r'$\tau (s)$')
        plt.ylabel('MSD (m^2)')
        plt.show()
        plt.errorbar(tau,MSDy,unc_y)
        plt.title('Y MSD')
        plt.xlabel(r'$\tau (s)$')
        plt.ylabel('MSD (m^2)')
        plt.show()
        # plot msds (cutoff)
        xcut = float(input("Enter x cutoff (s):"))
        ycut = float(input("Enter y cutoff (s):"))
        Nx = int(xcut*np.ceil(FPS))
        Ny = int(ycut*np.ceil(FPS))
        plt.errorbar(tau[0:Nx],MSDx[0:Nx],unc_x[0:Nx])
        plt.title('X MSD')
        plt.xlabel(r'$\tau (s)$')
        plt.ylabel('MSD (m^2)')
        plt.show()
        plt.errorbar(tau[0:Ny],MSDy[0:Ny],unc_y[0:Ny])
        plt.title('Y MSD')
        plt.xlabel(r'$\tau (s)$')
        plt.ylabel('MSD (m^2)')
        plt.show()
        # now regress
        resx = sp.optimize.curve_fit(examplefunc,tau[4:N],MSDx[4:N],sigma=unc_x[4:N],p0=3e-14,absolute_sigma=True)
        resy = sp.optimize.curve_fit(examplefunc,tau[4:N],MSDy[4:N],sigma=unc_y[4:N],p0=3e-14,absolute_sigma=True)
        m_x = resx[0]
        m_y = resy[0]
        m_uncx = np.sqrt(resx[1])/m_x
        m_uncy = np.sqrt(resy[1])/m_y
        # propagate errors
        m_uncx = np.sqrt(m_uncx**2 + (dc/c)**2)
        m_uncy = np.sqrt(m_uncy**2 + (dc/c)**2)
        # find radii
        Rx = (2*kB*T)/(3*np.pi*m_x*vx)
        Ry = (2*kB*T)/(3*np.pi*m_y*vy)
        # propagate error
        Rx_er = np.sqrt((m_uncx)**2 + (vx_er/vx)**2)
        Ry_er = np.sqrt((m_uncy)**2 + (vy_er/vy)**2)
        # print results
        print("X Radius: ",Rx,"pm",Rx_er*Rx)
        print("Y Radius: ",Ry,"pm",Ry_er*Ry)
        # plot results
        fxvals = examplefunc(tau,m_x)
        fyvals = examplefunc(tau,m_y)
        plt.plot(tau[0:Nx],MSDx[0:Nx],'r.')
        plt.plot(tau[0:Nx],fxvals[0:Nx])
        plt.xlabel(r'$\tau (s)$')
        plt.ylabel('MSD (m^2)')
        plt.title("X Fit")
        plt.show()
        plt.title("Y Fit")
        plt.xlabel(r'$\tau (s)$')
        plt.ylabel('MSD (m^2)')
        plt.plot(tau[0:Ny],MSDy[0:Ny],'r.')
        plt.plot(tau[0:Ny],fyvals[0:Ny])
        plt.show()
    return 0

main()