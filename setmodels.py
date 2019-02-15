import numpy
import scipy
import astropy
import hmf
import sys
import os
import argparse
from scipy.interpolate import CubicSpline
from astropy.cosmology import default_cosmology
from colossus.cosmology import cosmology
from colossus.halo import concentration
from colossus.lss import bias

global pn

# parameter names
pn = [
    'COLOCOSMO', 'HMFCOSMO', 'HMFMODEL',
    'BIASMODEL', 'CONCMODEL', 'MASSDEF',
    'M_min', 'M_max', 'z_min',
    'z_max', 'k_min', 'k_max',
    'bin_numb'
]

def main():

    if len(sys.argv) != 2:
        sys.exit('Need parameter file\nTeminating...')

    params = ReadParams(sys.argv[1])
    
    # set up colossus' cosmology model
    cosmo = cosmology.setCosmology(params[pn[0]])
    
    # set up hmf's cosmology model
    hmf_cosmo_model = default_cosmology.get_cosmology_from_string(params[pn[1]])
    hmf_cosmo = hmf.cosmo.Cosmology(cosmo_model=hmf_cosmo_model)
    
    # set the seperation for hmf module
    Mdef     = params[pn[5]] # definition of halo mass
    M_min    = params[pn[6]]
    M_max    = params[pn[7]]
    bin_numb = params[pn[-1]].astype(numpy.int)
    dlog10m  = (M_max - M_min) / bin_numb 
    
    # set up redshift & wavenumber array
    #m = numpy.linspace(numpy.power(10.,M_min),numpy.power(10.,M_max),bin_numb)
    z = numpy.linspace(params[pn[8]],  params[pn[9]],  bin_numb)
    k = numpy.linspace(params[pn[10]], params[pn[11]], bin_numb)
    
    # initialize
    dndm_z = numpy.empty((bin_numb, bin_numb))
    P_lin  = numpy.empty((bin_numb, bin_numb))
    conc   = numpy.empty((bin_numb, bin_numb))
    Bias   = numpy.empty((bin_numb, bin_numb))
    
    
    for idx_z in range(bin_numb):
        # halo mass function
        HaloMassModule = hmf.hmf.MassFunction(
            z=z[idx_z], Mmin=M_min, Mmax=M_max,
            dlog10m=dlog10m, hmf_model=params[pn[2]]
        )
        #dndm = HaloMassModule.dndm
        #mass = HaloMassModule.m
        #spline_dndm = CubicSpline(mass, dndm)
        #dndm_z[idx_z] = spline_dndm(m)
        dndm_z[idx_z] = HaloMassModule.dndm
        if idx_z == 0: m = HaloMassModule.m
    
        # concentration
        conc[idx_z] = concentration.concentration(m, Mdef, z[idx_z],
                                                  model=params[pn[4]])
    
        # linear power spectrum
        P_lin[idx_z] = cosmo.matterPowerSpectrum(k, z[idx_z])
    
        # bias
        Bias[idx_z] = bias.haloBias(m, z[idx_z], mdef=Mdef,
                                    model=params[pn[3]])
    
    from astropy.io import fits
    
    col_m = fits.Column(name='m200', array=m, format='E')
    col_z = fits.Column(name='z'   , array=z, format='E')
    col_k = fits.Column(name='k'   , array=k, format='E')
    
    matrix_format = '%dE' % bin_numb
    col_dndm = fits.Column(name='dndm', array=dndm_z, format=matrix_format)
    col_conc = fits.Column(name='conc', array=conc  , format=matrix_format)
    col_Plin = fits.Column(name='Plin', array=P_lin , format=matrix_format)
    col_bias = fits.Column(name='bias', array=Bias  , format=matrix_format)
    coldefs = fits.ColDefs([col_m, col_z, col_k,
                            col_dndm, col_conc,
                            col_Plin, col_bias])
    table = fits.BinTableHDU.from_columns(coldefs)
    table.writeto('DM_model.fits')


def CheckParams(dicts):
    LostParams = []
    for param in pn:
        if param in dicts:
            pass
        else:
            LostParams.append(param)

    if LostParams:
        print("Missing Parameter(s): ", LostParams)
        sys.exit("Terminating the code...")
    else:
        return

def ReadParams(filename):
    param_dict = {}
    param_set = numpy.loadtxt(filename, dtype=numpy.str)
    for idx in range(param_set.shape[0]):
        if not param_set[idx,0].isupper():
            param_dict.update(
                    {param_set[idx,0]:
                     param_set[idx,1].astype('f') }
            )
        else:
            param_dict.update(
                    {param_set[idx,0]:
                     param_set[idx,1] }
            )
    CheckParams(param_dict)
    return param_dict

if __name__ == '__main__':
    main()
