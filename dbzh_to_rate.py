import numpy as np
import math
from pyproj import CRS
from pyproj import Transformer
import json


def read_coef(configfile="config_dbzhtorate.json"):
    """ Read Z-R and Z-S conversion coefficients from file.
    
    Keyword arguments:
    configfile -- json file containing coefficients

    Return:
    coef -- dictionary containing coefficients
    """
    
    with open(configfile, "r") as jsonfile:
        data = json.load(jsonfile)
 
    coef = data['coef']

    return coef


def dBZtoRR(dbz, coef):
    """ Convert dBZ to rain rate (frontal/convective rain).

    Keyword arguments:
    dbz -- Array of dBZ values
    coef -- dictionary containing Z(R) A and B coefficients zr_a, zr_b, zr_a_c and zr_a_c (c for convective rain)

    Return:
    rr -- rain rate
    """
    
    zr_a = coef['zr_a']
    zr_b = coef['zr_b']
    zr_a_c = coef['zr_a_c']
    zr_b_c = coef['zr_b_c']

    # Calculate dBZ limit when to use frontal/convective rain rate formula
    if zr_a == zr_a_c:
        conv_dbzlim = 10.0 * math.log10(zr_a)
    else:
        R = (zr_a / zr_a_c) ** (1.0 / (zr_b_c - zr_b))
        conv_dbzlim = 10.0 * math.log10(zr_a * (R ** zr_b))
        
    #Convert dBZ to rain rate RR
    idx = dbz < conv_dbzlim
    rr = np.where(idx, 10 ** (dbz / (10 * zr_b) + (-math.log10(zr_a) / zr_b)), 10 ** (dbz / (10 * zr_b_c) + (-math.log10(zr_a_c) / zr_b_c)))

    return rr


def dBZtoSR(dbz, coef):
    """ Convert dBZ to snow rate.

    Keyword arguments:
    dbz -- Array of dBZ values
    coef -- dictionary containing Z(R) A and B coefficients

    Return:
    sr -- snow rate
    """

    zs_a = coef['zs_a']
    zs_b = coef['zs_b']

    sr = 10 ** (dbz / (10 * zs_b) + (-math.log10(zs_a) / zs_b))

    return sr


def calc_lookuptables_dBZtoRATE(timeresolution, coef, nodata, undetect, gain, offset):
    """ Calculate look-up tables for dBZ to RR and dBZ to SR conversion.

    Keyword arguments:
    timeresolution -- 
    coef -- dictionary containing Z(R) A and B coefficients zr_a, zr_b, zr_a_c and zr_a_c (c for convective rain)

    Return:
    lut_rr -- look-up table for rain rate conversion
    lut_sr -- look-up table for snow rate conversion
    """

    # Get coefficients for Z-R and Z-S conversion
    zr_a = coef['zr_a']
    zr_b = coef['zr_b']
    zr_a_c = coef['zr_a_c']
    zr_b_c = coef['zr_b_c']
    zs_a = coef['zs_a']
    zs_b = coef['zs_b']    

    # Make look up table
    dbz = np.array(range(0,nodata+1))
    dbz = dbz * gain + offset

    # Calculate values for look-up tables and convert to mm/timeresolution
    lut_rr = dBZtoRR(dbz, coef)
    lut_rr = lut_rr / (60 / timeresolution)
    
    lut_sr = dBZtoSR(dbz, coef)
    lut_sr = lut_sr / (60 / timeresolution)

    #Nodata and undetect
    lut_rr[undetect] = -0.0001
    lut_rr[nodata] = 0

    lut_sr[undetect] = -0.0001
    lut_sr[nodata] = 0
    
    return lut_rr, lut_sr


def dBZtoRATE_lut(dbz, lut_rr, lut_sr, snowprob):
    """ Convert dBZ to rate using look-up table.

    Keyword arguments:
    dbz -- dBZ field
    lut_rr -- look-up table for dBZ - rain rate conversion
    lut_sr -- look-up table for dBZ - snow rate conversion
    snowprob -- probability of snow array (values from 0 to 100)

    Return:
    rate -- precipitation rate

    """
    # Calculate rain rate
    rate = lut_rr[dbz]
    rate[snowprob > 50] = lut_sr[dbz][snowprob > 50]

    return rate


def dBZtoRR_lut(dbz, lut):
    """ Convert dBZ to rate using look-up table.

    Keyword arguments:
    dbz -- dBZ field
    lut -- look-up table for dBZ - rate conversion

    Return:
    rr -- rain rate

    """

    rr=lut[dbz]
    return rr
