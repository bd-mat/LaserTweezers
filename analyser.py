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

derr = np.sqrt(derr**2 + (c*std)**2)
derr += c*std

# apply 
# calculate cutoff
cutoff = 1 # s
N = int(cutoff*np.ceil(FPS))

FILE1 = 'brown2'
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
        if n % 100 == 0:
            print(str(int(100*n/(N-1)))+'% done.')
    return MSD,unc

def parabola(x,a):
    return 0.5*a*(x**2)

def potential(data,res,unc,lbl):
    hist = np.histogram(data,bins=res)
    print(res, "bins")
    hvals = np.array(hist[0])
    hunc = np.sqrt(hvals)
    maxval = hvals.max()
    hvals = hvals/(maxval)
    hunc = hunc/(maxval)
    xspac = np.array(hist[1])
    xavg = np.zeros_like(hvals)
    for i in range(0,res):
        xavg[i] = (xspac[i]+xspac[i+1])/2
    U = np.zeros(res,dtype=float)
    Uunc = np.zeros_like(U)
    for i in range(0,res):
        if hvals[i]==0:
            pass
        else:
            U[i] = -(kB*T)*np.log(hvals[i])
            Uunc[i] = (kB*T)*(hunc[i]/hvals[i])
    plt.plot(xavg,U)
    plt.show()
    bvals = np.linspace(0,res,res)
    plt.errorbar(bvals,U,Uunc)
    plt.show()
    low = int(input("Input lower bound (bin index)"))
    high = int(input("Input upper bound (bin index)"))
    res = sp.optimize.curve_fit(parabola,xavg[low:high],U[low:high],
                                sigma=Uunc[low:high],absolute_sigma=True,p0=1e-06)
    plt.errorbar(xavg[low:high],U[low:high],Uunc[low:high],fmt='xr')
    fvals = parabola(xavg,res[0])
    kunc = np.sqrt(res[1])
    plt.plot(xavg[low:high],fvals[low:high])
    plt.title("Potential vs "+lbl+" displacement")
    plt.ylabel("U (J)")
    plt.xlabel("x (m)")
    fig = plt.gcf()
    #chi2 = sp.stats.chisquare(U[low:high],fvals[low:high],high-low-1)
    plt.show()
    print("Kappa value:",res[0])
    print("Uncertainty:",kunc)
    check = input("Save potential plot?")
    if check == 'y':
        fig.savefig(FILE1+"_" + lbl + "potential.png",dpi=300)
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

check = input('Run potential fitting?')
if check == 'y':
    resol = int(input("Number of bins:"))
    potential(xvals,resol,derr,'x')
    potential(yvals,resol,derr,'y')

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

derr = c/2

check = input('Calculate MSDs?')
if check == 'y':
    
    MSD_resx = MSD(xvals,derr)
    MSD_resy = MSD(yvals,derr)
    MSDx = MSD_resx[0]
    MSDy = MSD_resy[0]
    unc_x = MSD_resx[1]
    unc_y = MSD_resy[1]
    
    tau = (1/FPS)*np.linspace(0,len(MSDx)-1,num=len(MSDx))
    

    plt.plot(tau,MSDx)
    plt.show()
    xcut = float(input("X cutoff? (in seconds)"))
    Nx = int(xcut*np.ceil(FPS))
    plt.plot(tau,MSDy)
    plt.show()
    ycut = float(input("Y cutoff?"))
    Ny = int(ycut*np.ceil(FPS))
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
    resx = sp.optimize.curve_fit(examplefunc,tau[1:Nx],MSDx[1:Nx],sigma=unc_x[1:Nx],p0=3e-14,absolute_sigma=True)
    resy = sp.optimize.curve_fit(examplefunc,tau[1:Ny],MSDy[1:Ny],sigma=unc_y[1:Ny],p0=3e-14,absolute_sigma=True)
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
    
    fvalsx = examplefunc(tau,m_x)
    fvalsy = examplefunc(tau,m_y)
    
    plt.plot(tau[0:Nx],MSDx[0:Nx],label='MSD')
    plt.plot(tau[0:Nx],MSDx[0:Nx]+unc_x[0:Nx],'--k',label='Range of Uncertainty')
    plt.plot(tau[0:Nx],MSDx[0:Nx]-unc_x[0:Nx],'--k')
    plt.plot(tau[0:Nx],fvalsx[0:Nx],'r',label='Fit')
    plt.legend()
    plt.plot()
    plt.title('X MSD')
    plt.xlabel(r'$\tau (s)$')
    plt.ylabel('MSD (m^2)')
    xplot = plt.gcf()
    plt.show()
    plt.plot(tau[0:Ny],MSDy[0:Ny],label='MSD')
    plt.plot(tau[0:Ny],MSDy[0:Ny]+unc_y[0:Ny],'--k',label='Range of Uncertainty')
    plt.plot(tau[0:Ny],MSDy[0:Ny]-unc_y[0:Ny],'--k')
    plt.plot(tau[0:Ny],fvalsy[0:Ny],'r',label='Fit')
    plt.legend()
    plt.title('Y MSD')
    plt.xlabel(r'$\tau (s)$')
    plt.ylabel('MSD (m^2)')
    yplot = plt.gcf()
    plt.show()
    check1 = input('Save plots?')
    if check1 == 'y':
        xplot.savefig(FILE1+'_xMSD.png',dpi=300)
        yplot.savefig(FILE1+'_yMSD.png',dpi=300)
    