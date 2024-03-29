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


def calc_conv_dBZlim(coef):
    """ Calculate dBZ limit for convective/frontal case Z-R calculation.
    Limit with default values is 23.48 dBZ.
    
    Keyword arguments:
    coef -- dictionary containing Z(R) A and B coefficients zr_a, zr_b, zr_a_c and zr_a_c (c for convective rain)

    Return:
    conv_dbzlim -- Limit dBZ value for convective rain rate calculation
    """

    zr_a = coef['zr_a']
    zr_b = coef['zr_b']
    zr_a_c = coef['zr_a_c']
    zr_b_c = coef['zr_b_c']
    
    if zr_a == zr_a_c:
        conv_dbzlim = 10.0 * math.log10(zr_a)
    else:
        R = (zr_a / zr_a_c) ** (1.0 / (zr_b_c - zr_b))
        conv_dbzlim = 10.0 * math.log10(zr_a * (R ** zr_b))

    return conv_dbzlim



def dBZtoRR(dbz, coef, conv_dbzlim):
    """ Convert dBZ to rain rate (frontal/convective rain).

    Keyword arguments:
    dbz -- Array of dBZ values
    coef -- dictionary containing Z(R) A and B coefficients zr_a, zr_b, zr_a_c and zr_a_c (c for convective rain)
    conv_dbzlim -- dBZ limit, when to use frontal/convective rain rate formula

    Return:
    rr -- rain rate
    """
    
    zr_a = coef['zr_a']
    zr_b = coef['zr_b']
    zr_a_c = coef['zr_a_c']
    zr_b_c = coef['zr_b_c']
    
    #Help coefficients for rain
    zr_b_10 = zr_b_10_c = 10.0 * zr_b
    zr_c = zr_c_c = -math.log10(zr_a) / zr_b

    if zr_a_c > 0.0 and zr_b_c > 0.0:
        zr_b_10_c = 10.0 * zr_b_c
        zr_c_c = -math.log10(zr_a_c) / zr_b_c

    #Convert dBZ to rain rate RR
    idx = dbz < conv_dbzlim
    rr = np.where(idx, 10 ** (dbz / (zr_b_10) + zr_c), 10 ** (dbz / (zr_b_10_c) + zr_c_c))

    return rr



def calc_lookuptable_dBZtoRR(timeresolution, coef, nodata, undetect, gain, offset):
    """ Calculate look-up table for dBZ to RR conversion.

    Keyword arguments:
    timeresolution -- 
    coef -- dictionary containing Z(R) A and B coefficients zr_a, zr_b, zr_a_c and zr_a_c (c for convective rain)

    Return:
    lut -- look-up table
    """

    # Get coefficients for Z-R conversion
    zr_a = coef['zr_a']
    zr_b = coef['zr_b']
    zr_a_c = coef['zr_a_c']
    zr_b_c = coef['zr_b_c']

    # Make look up table
    dbz = np.array(range(0,nodata+1))
    dbz = dbz * gain + offset
    conv_dbzlim = calc_conv_dBZlim(coef)
    lut = dBZtoRR(dbz, coef, conv_dbzlim)

    #Convert to mm/timeresolution
    lut = lut / (60 / timeresolution)

    #Nodata and undetect
    lut[undetect] = -0.0001
    lut[nodata] = 0
    
    return lut



def dBZtoRR_lut(dbz,lut):
    """ Convert dBZ to rate using look-up table.

    Keyword arguments:
    dbz -- dBZ field
    lut -- look-up table for dBZ - rate conversion

    Return:
    rr -- rain rate

    """

    rr=lut[dbz]
    return rr



def dBZtoRate(dbz, coef, conv_dbzlim):
    """ Convert dBZ to precipitation (frontal/convective rain or snow).

    Keyword arguments:
    dbz -- Array of dBZ values
    coef -- dictionary containing Z(R) A and B coefs zr_a, zr_b, zr_a_c and zr_a_c (c for convective rain) and Z(S) coefs zs_a and zs_b
    conv_dbzlim -- dBZ limit, when to use frontal/convective rain rate formula

    Return:
    rr -- rain rate
    sr -- snow rate

    """

    zr_a = coef['zr_a']
    zr_b = coef['zr_b']
    zr_a_c = coef['zr_a_c']
    zr_b_c = coef['zr_b_c']
    zs_a = coef['zs_a']
    zs_b = coef['zs_b']
    
    #Help coefficients for rain
    zr_b_10 = zr_b_10_c = 10.0 * zr_b
    zr_c = zr_c_c = -math.log10(zr_a) / zr_b
    if zr_a_c > 0.0 and zr_b_c > 0.0:
        zr_b_10_c = 10.0 * zr_b_c
        zr_c_c = -math.log10(zr_a_c) / zr_b_c

    #Convert dBZ to rain rate RR
    idx = dbz < conv_dbzlim
    rr = np.where(idx, 10 ** (dbz / zr_b_10 + zr_c), 10 ** (dbz / zr_10_c + zr_c_c))

    #Help coefficients for snow
    zs_b_10 = zs_b_10_c = 10.0 * zs_b
    zs_c = zs_c_c = -math.log10(zs_a) / zs_b

    #Convert dBZ to snow rate SR
    sr = dbz / zs_b_10 + zs_c
   
    # To be done: Snow rate calculation
    # For sleet: currently done by weighting snow or rain rate formula, does not
    # take into account the bright band

    return rr, sr
