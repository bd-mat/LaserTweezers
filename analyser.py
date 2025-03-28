# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import random as ran
import scipy as sp

T = 296 # Kelvin
kB = 1.381e-23 # J/K
c = 0.091e-6 # m/pixel
dc = 0.001e-6 # c uncertainty

# error on a single measurement is c/2
derr = c/2

# radius
R = 1.55e-6 # radius of bead, m
FPS = 310.1

if FPS==310.1:
    std = 0.6425
else:
    std = 0.485

derr += c*std

# apply 
# calculate cutoff
cutoff = 1 # s
N = int(cutoff*np.ceil(FPS))

FILE1 = '90'
FILE2 = '_vals.csv'

def random_walk(N):
    x = np.zeros(N)
    for i in range(1,N):
        if ran.random() <= 0.5:
            x[i] = x[i-1] + 1
        else:
            x[i] = x[i-1] - 1
    return x

def read_data(filename):
    data = np.genfromtxt(filename,dtype=float,delimiter=',',skip_header=True)
    return data

def savefigs(tvals,xvals,yvals,FILE1):
    plt.plot(tvals,xvals)
    plt.title('X displacement vs time step')
    plt.xlabel('Time step')
    plt.ylabel('Displacement (m)')
    plt.savefig(FILE1+'_xplot',dpi=300)
    plt.show()
    plt.plot(tvals,yvals)
    plt.title('Y displacement vs time step')
    plt.xlabel('Time step')
    plt.ylabel('Displacement (m)')
    plt.savefig(FILE1+'_yplot',dpi=300)
    plt.show()

def gaussian(data,res,name='Gaussian'):
    plt.hist(data,bins=(1/res)*np.arange(res*data.min(), res*data.max()+1))
    plt.title(name)
    plt.show()
    
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

def parabola(x,a):
    return 0.5*a*(x**2)

def potential(data,res):
    hist = np.histogram(data,bins=res)
    print(res, "bins")
    hvals = np.array(hist[0])
    hvals = hvals/(hvals.max())
    xspac = np.array(hist[1])
    xavg = np.zeros_like(hvals)
    for i in range(0,res):
        xavg[i] = (xspac[i]+xspac[i+1])/2
    U = np.zeros(res,dtype=float)
    for i in range(0,res):
        if hvals[i]==0:
            pass
        else:
            U[i] = -(kB*T)*np.log(hvals[i])
    plt.plot(xavg,U)
    plt.show()
    bvals = np.linspace(0,res,res)
    plt.plot(bvals,U)
    plt.show()
    low = int(input("Input lower bound (bin index)"))
    high = int(input("Input upper bound (bin index)"))
    res = sp.optimize.curve_fit(parabola,xavg[low:high],U[low:high],p0=1e-06)
    plt.plot(xavg[low:high],U[low:high],'xr')
    fvals = parabola(xavg,res[0])
    plt.plot(xavg[low:high],fvals[low:high])
    plt.title("Potential vs x displacement")
    plt.ylabel("U (J)")
    plt.xlabel("x (m)")
    fig = plt.gcf()
    plt.show()
    print("Kappa value:",res[0])
    check = input("Save potential plot?")
    if check == 'y':
        fig.savefig(FILE1+"_potential_plot.png",dpi=300)
    return xavg,hist
    

def examplefunc(x,a):
    return a*x


data = read_data(FILE1+FILE2)
tvals = data[:,0]
xvals = data[:,1]
yvals = data[:,2]

# convert to SI
tvals = tvals*(1/FPS)
xvals = xvals*c
yvals = yvals*c

xunc = derr/xvals
yunc = derr/xvals

xmean = np.mean(xvals)
ymean = np.mean(yvals)

xvals = xvals - xmean
yvals = yvals - ymean

resol = int(input("Number of bins:"))
potential(xvals,resol)

# find error on each value

check = input('Generate and save plots? (y for yes)')
if check == 'y':
    savefigs(tvals,xvals,yvals,FILE1)

check = input("Find kappa (old method)?")
if check == 'y':
    xvar = np.var(xvals)
    yvar = np.var(yvals)
    K_x = (kB*T)/(xvar)
    K_y = (kB*T)/(yvar)
    print('K_x: ',K_x, " ----- K_y: ",K_y)

#  plot gaussian
check = input('Plot gaussians?')
if check == 'y':
    gaussian(xvals/c,20,'X Gaussian')
    gaussian(yvals/c,20,'Y Gaussian')

# Now find the MSD ! :)

# test random walk
    
check = input('Random walk?')
if check == 'y':
    randx = random_walk(1000)
    randy = random_walk(1000)
    plt.plot(randx,randy)
    plt.show()
    
    MSDr = MSD(randx)
    taur = np.linspace(0,999,num=999,endpoint=0)
    
    plt.plot(taur[0:400],MSDr[0:400])
    plt.show()

# now the real thing

check = input('Calculate MSDs?')
if check == 'y':
    
    MSD_resx = MSD(xvals,derr)
    MSD_resy = MSD(yvals,derr)
    MSDx = MSD_resx[0]
    MSDy = MSD_resy[0]
    unc_x = MSD_resx[1]
    unc_y = MSD_resy[1]
    
    tau = (1/FPS)*np.linspace(0,len(MSDx)-1,num=len(MSDx))

    plt.errorbar(tau[0:N],MSDx[0:N],unc_x[0:N])
    plt.title('X MSD')
    plt.xlabel(r'$\tau (s)$')
    plt.ylabel('MSD (m^2)')
    plt.show()
    plt.errorbar(tau[0:N],MSDy[0:N],unc_y[0:N])
    plt.title('Y MSD')
    plt.xlabel(r'$\tau (s)$')
    plt.ylabel('MSD (m^2)')
    plt.show()
    check2 = input('Save data?')
    if check2 == 'y':
        file = open(FILE1+'_MSD.csv','w')
        file.write('n,MSDx,MSDy\n')
        for i in range(0,len(MSDx)):
            file.write(str(i)+','+str(MSDx[i])+","+str(MSDy[i])+"\n")
        file.close()

# now calculate the viscosity
# rearrange for exponential
if check =='y':
    # old stuff
    # MSD_adj_x = -np.log(1-(K_x/(2*kB*T))*MSDx[0:N])
    # MSD_adj_y = -np.log(1-(K_y/(2*kB*T))*MSDy[0:N])
    # plt.plot(tau[0:N],MSD_adj_x)
    # plt.plot(tau[0:N],MSD_adj_y)
    # plt.show()
    
    # regress with error
    resx = sp.optimize.curve_fit(examplefunc,tau[1:N],MSDx[1:N],sigma=unc_x[1:N],p0=3e-14,absolute_sigma=True)
    resy = sp.optimize.curve_fit(examplefunc,tau[1:N],MSDy[1:N],sigma=unc_y[1:N],p0=3e-14,absolute_sigma=True)
    m_x = resx[0]
    m_y = resy[0]
    m_uncx = np.sqrt(resx[1])/m_x
    m_uncy = np.sqrt(resy[1])/m_y
    # propagate errors
    m_uncx = np.sqrt(m_uncx**2 + (dc/c)**2)
    m_uncy = np.sqrt(m_uncy**2 + (dc/c)**2)
    print(resx)
    print(resy)
    """
    # regress
    resx = sp.stats.linregress(tau[0:N],MSDx[0:N])
    resy = sp.stats.linregress(tau[0:N],MSDy[0:N])
    # find pc. unc
    tcx_unc = resx[4]/resx[0]
    tcy_unc = resy[4]/resy[0]
    # propagate
    vcx_unc = np.sqrt(tcx_unc**2 + (dc/c)**2)
    vcy_unc = np.sqrt(tcy_unc**2 + (dc/c)**2)
    m_x = (resx[0])
    m_y = (resy[0])
    """
    visc_x = (2*kB*T)/(3*np.pi*R*m_x)
    visc_y = (2*kB*T)/(3*np.pi*R*m_y)
    print('X Viscosity: ', visc_x*(1e3), ' pm ', m_uncx*visc_x*(1e3), 'mPa')
    print('Y Viscosity: ', visc_y*(1e3), ' pm ', m_uncy*visc_y*(1e3), 'mPa')
    