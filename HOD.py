import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative
from scipy.special import erf, sici, j0
from scipy.integrate import romb, cumtrapz, trapz
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from colossus.cosmology import cosmology
from astropy.io import fits
import hmf
import sys, time

global mycosmo, mdef
global Verbose

mycosmo = cosmology.setCosmology('planck15')
mdef  = '200m'
Verbose = False

def AngCorrFuncModel(theta, dvar, HODparams):
    m = dvar.field(0)
    z = dvar.field(1)
    k = dvar.field(2)

    rz = mycosmo.comovingDistance(0., z, transverse=False)
    rzdz_spline = IUS(z, rz, k=3)
    rzdz = rzdz_spline.derivative(n=1)
    dzrz = 1. / rzdz(z)
    #print('dzrz',dzrz)

    # zeroBessel_1(k, z, theta)
    zk_matrix = k * z.reshape(-1,1)
    zeroBessel_1 = j0(oneto3d(theta) * BCtoCube(zk_matrix))

    k_Integrand = k * GalaxPowerSpec(dvar, HODparams) * zeroBessel_1 / np.pi
    k_Integral  = cumtrapz(k_Integrand, k, axis=2, initial=0)[:,-1]

    #z_Integrand = NormNz * NormNz * dzrz * k_Integral
    z_Integrand = dzrz * k_Integral
    omega_theta  = cumtrapz(z_Integrand, z, axis=1, initial=0)[:,-1]

    return omega_theta

def GalaxPowerSpec(dvar, HODparams):

    return OneHalo(dvar, HODparams) +\
           TwoHalo(dvar, HODparams)

def OneHalo(dvar, HODparams):
    return CS_OneHalo(dvar, HODparams) +\
           SS_OneHalo(dvar, HODparams)

def CS_OneHalo(dvar, HODparams):
    m = dvar.field(0)
    z = dvar.field(1)
    k = dvar.field(2)

    CenN, SatN = OccupationNumber(m, HODparams)
    NcNs = CenN * SatN
    coeff = 2. / np.power(NumberDensity(detm_var, HODparams), 2.)

    FTNNFW = FT_NormNFW(dvar)

    # dvar.field(3) -> dndm
    Integrand = Transp(coeff) * NcNs * dvar.field(3) * FTNNFW
    P_cs      = cumtrapz(Integrand, m, axis=2, initial=0)[:,-1]

    # return a (z, k) matrix
    return P_cs

def SS_OneHalo(dvar, HODparams):
    m = dvar.field(0)
    z = dvar.field(1)
    k = dvar.field(2)

    _, SatN = OccupationNumber(m, HODparams)
    NsNs = SatN * SatN
    coeff = 1. / np.power(NumberDensity(dvar, HODparams), 2.)

    FTNNFW = FT_NormNFW(dvar)

    # dvar.field(3) -> dndm
    Integrand = Transp(coeff) * NsNs * dvar.field(3) * FTNNFW * FTNNFW
    P_ss      = cumtrapz(Integrand, m, axis=2, initial=0)[:,-1]

    # return a (z, k) matrix
    return P_ss

def TwoHalo(dvar, HODparams):
    m = dvar.field(0)
    z = dvar.field(1)
    k = dvar.field(2)
    bias = dvar.field(6) 
    MatterPowerSpec = dvar.field(5)

    coeff = 1. / NumberDensity(dvar, HODparams)
    
    totN = MeanOccupationNumber(m, HODparams)

    FTNNFW = FT_NormNFW(dvar)

    # dvar.field(3) -> dndm
    Integrand = totN * dvar.field(3) * FTNNFW
    Integrand = Transp(coeff) * BCtoCube(bias) * Integrand

    biasTerm = cumtrapz(Integrand, m, axis=2, initial=0)[:,:,-1]

    return MatterPowerSpec.T * biasTerm * biasTerm 

def bias_file(dvar):
    return dvar['bias']

def Plin_file(dvar):
    return dvar['Plin']

def FT_NormNFW(dvar):
    m = dvar['m200']
    k = dvar['k']
    c = conc_file(dvar)

    rhos  = rho_s(dvar)
    rs    = r_s(dvar)

    coeff = 4. * np.pi * rhos * np.power(rs, 3.) / m

    krs   = oneto3d(k) * BCtoCube(rs)
    krs1c = krs * (1+c)

    sinkrs = np.sin(krs)
    coskrs = np.cos(krs)

    sinInteg_krs  , cosInteg_krs   = sici(krs)
    sinInteg_krs1c, cosInteg_krs1c = sici(krs1c)
    deltaSinI = sinInteg_krs - sinInteg_krs1c
    deltaCosI = cosInteg_krs - cosInteg_krs1c

    term_1st = sinkrs * deltaSinI
    term_2nd = np.sin(c * krs) / ((1 + c) * krs)
    term_3rd = coskrs * deltaCosI
    mainpart = (term_1st + term_2nd + term_3rd)

    return coeff * mainpart

def r_s(dvar):
    m = dvar['m200']
    z = dvar['z']
    r = m / (rho_s(dvar) * fz(c) * np.pi * 4.)
    return np.power(r, 1./3.)

def rho_s(dvar):
    rho_c = mycosmo.rho_c(dvar['z'])
    c = conc_file(dvar)
    delta_c =  200. * np.power(c, 3.) / 3. / fz(c)
    return Transp(rho_c) * delta_c

def fz(c):
    return np.log(1.+c) + c/(1.+c)

def conc_file(dvar):
    # Model should be a fit_record object
    return dvar['conc']

def conc(m, z):
    from colossus.halo import concentration
    # adding 'model' argument to change the
    # default setting, diemer19.
    for idx_z in range(z.shape[0]):
        tmp = concentration.concentration(m, mdef, z[idx_z])
        if idx_z == 0:
            c = tmp
        else :
            c = np.vstack((c, tmp))
    return c

def concc():
    c = 5.
    return c

def NumberDensity(dvar, HODparams):
    M    = dvar['m200']
    z    = dvar['z']
    dndm = dvar['dndm']
    N    = MeanOccupationNumber(M, HODparams)
    Ndensity_z3 = cumtrapz(dndm*N, M, axis=1)[:,-1]
    print('cumtrapz',Ndensity_z3)
    Ndensity_z2 = trapz(dndm*N, M, axis=1)
    print('trapz',Ndensity_z3)
    dndlogm=np.log(10.)*M*dndm
    dx = np.log10(M[1])-np.log10(M[0])
    dx = M[1]-M[0]
    Ndensity_z1 = romb(dndm*N, dx=dx, axis=1)
    print('romb',Ndensity_z1)
    print('difference between romp & trapz', (Ndensity_z2-Ndensity_z1))


    return Ndensity_z3

def MeanOccupationNumber(M, HODparams):
    CenN, SatN = OccupationNumber(M, HODparams)
    return CenN + SatN

def OccupationNumber(M, HODparams):
    logMmin, Msigma, Mcut, Msat, alpha = HODparams
    # central component
    Central = (1 + erf((np.log10(M) - logMmin) / Msigma)) / 2.
    # satellite component
    ratio_M = (M-Mcut)/ Msat
    Satellite = Central * np.power(ratio_M, alpha)

    return Central, Satellite

def Mcut(logMmin):
    return np.power(np.power(10., logMmin), -0.5)

def HMF_file(dvar):
    return dvar('m200'), dvar('dndm')

def BCtoCube(array):
    size = array.shape[0]
    return np.broadcast_to(array, (size, size, size))

def oneto3d(array):
    return array[:,np.newaxis,np.newaxis]

def Transp(array):
# this function is mainly for transposing
# 1-d numpy array
    return array.reshape(-1, 1)

def ReadModel():
    #colnames = ['m200', 'z', 'k', 'dndm', 'conc', 'Plin', 'bias']
    ModelFits = fits.open('DM_model.fits')
    det_variables = ModelFits[1].data
    return det_variables

if __name__ == '__main__':

    logMmin = 13.
    Msigma  = 0.2
    logMsat = 12.51
    Msat    = np.power(10., logMsat)
    M_cut   = Mcut(logMmin)
    alpha   = 1.0

    HODparams = [logMmin, Msigma, M_cut, Msat, alpha]

    detm_var = ReadModel()

    m = detm_var.field(0)
    print("%f"%((m[1]-m[0])/(m[2]-m[1])))

    CenN, SatN = OccupationNumber(m, HODparams)
    if Verbose: print('Central', CenN.shape)
    if Verbose: print('Satellite', SatN.shape)
    
    totN = MeanOccupationNumber(m, HODparams)
    if Verbose: print('Mean Occupation Number', totN.shape)

    numdens = NumberDensity(detm_var, HODparams)
    if Verbose: print('Number density', numdens)

    #c = conc_file(detm_var)
    #if Verbose: print('Concentration c(M200m, z)', c)

    #if Verbose: print('f(c)', fz(c))

    #u = FT_NormNFW(detm_var)
    #if Verbose: print('u matrix', u)

    #TwoHalo(detm_var, HODparams)

    #OneHalo(detm_var, HODparams)

    #theta = np.linspace(0., 1., m.shape[0])
    #omega = AngCorrFuncModel(theta, detm_var, HODparams) 
    #print(omega)
