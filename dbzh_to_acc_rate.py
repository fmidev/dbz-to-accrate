import h5py
import hiisi
import numpy as np
import argparse
import datetime
import configparser
import re
import gzip
from PIL import Image
import math
import time
import json

import dbzh_to_rate


def read_config(config_file):
    """ Read parameters from config file.
    
    Keyword arguments:
    config_file -- json file containing input parameters

    Return:
    coef -- dictionary containing coefficients
    input_conf -- dictionary containing input parameters
    output_conf -- dictionary containing output parameters

    """
    
    with open(config_file, "r") as jsonfile:
        data = json.load(jsonfile)
 
    coef = data['coef']
    input_conf = data['input']
    output_conf = data['output']
    
    return coef, input_conf, output_conf


def read_hdf5(image_h5_file):
    """ Read image array from ODIM hdf5 file.

    Keyword arguments:
    image_h5_file -- ODIM hdf5 file containing DBZH or RATE array
    
    Return:
    image_array -- numpy array containing DBZH or RATE array
    quantity -- array quantity, either 'DBZH' or 'RATE'
    timestamp -- timestamp of image_array
    mask_nodata -- masked array where image_array has nodata value
    gain -- gain of image_array
    offset -- offset of image_array

    """
    
    #Read RATE or DBZH from hdf5 file
    print('Extracting data from image h5 file')
    comp = hiisi.OdimCOMP(image_h5_file, 'r')
    #Read RATE array if found in dataset      
    test=comp.select_dataset('DBZH')
    if test != None:
        image_array=comp.dataset
        quantity='DBZH'
    else:
        #Look for RATE array
        test=comp.select_dataset('RATE')
        if test != None:
            image_array=comp.dataset
            quantity='RATE'
        else:
            print('Error: DBZH or RATE array not found in the input image file!')
            sys.exit(1)

    #Read nodata and undetect values from metadata for masking
    gen = comp.attr_gen('nodata')
    pair = gen.__next__()
    nodata = pair.value
    gen = comp.attr_gen('undetect')
    pair = gen.__next__()
    undetect = pair.value
    
    #Read gain and offset values from metadata
    gen = comp.attr_gen('gain')
    pair = gen.__next__()
    gain = pair.value
    gen = comp.attr_gen('offset')
    pair = gen.__next__()
    offset = pair.value

    #Read timestamp from metadata
    gen = comp.attr_gen('date')
    pair = gen.__next__()
    date = pair.value
    gen = comp.attr_gen('time')
    pair = gen.__next__()
    time = pair.value

    timestamp=date+time

    return image_array, quantity, timestamp, gain, offset, int(nodata), int(undetect)


def init_filedict_accumulation(image_h5_file):
    """ Copy common metadata from input file to be used as
    output file template.

    Keyword arguments:
    image_h5_file -- ODIM hdf5 file containing DBZH or RATE array

    Return:
    file_dict_accum -- dictionary containing ODIM metadata

    """
    
    #Read metadata from image h5 file
    comp = hiisi.OdimCOMP(image_h5_file, 'r')
    
    #Write metadata to file_dict
    file_dict_accum = {
        '/':dict(comp['/'].attrs.items()),
        '/how':dict(comp['/how'].attrs.items()),
        '/where':dict(comp['/where'].attrs.items())}

    return file_dict_accum


def write_accumulated_h5(output_h5,accumulated_image,file_dict_accum,date,time,startdate,starttime,enddate,endtime):
    """ Write accumulated precipitation rate to ODIM hdf5 file.
    
    Keyword arguments:
    output_h5 --
    accumulated_image --
    file_dict_accum --
    date --
    time --
    startdate --
    starttime --
    enddate --
    endtime --

    """
    
    #Insert date and time to file_dict
    file_dict_accum['/what'] = {
        'date':date,
        'object':np.string_("COMP"),
        'source':np.string_("ORG:247"),
        'time':time,
        'version':np.string_("H5rad 2.0")}
    #Insert startdate and -time and enddate- and time
    file_dict_accum['/dataset1/data1/what'] = {
        #'gain':0.001, # Onko tama oikein?
        #'nodata':65535,
        'gain':0,
        'nodata':-0.00001,
        'offset':0,
        'product':np.string_("COMP"),
        'quantity':np.string_("ACRR"),
        'undetect':0,
        'startdate':startdate,
        'starttime':starttime,
        'enddate':enddate,
        'endtime':endtime}
    #Insert accumulated dataset into file_dict
    file_dict_accum['/dataset1/data1/data'] = {
        'DATASET':accumulated_image,
        'COMPRESSION':'gzip',
        'COMPRESSION_OPTS':6,
        'CLASS':np.string_("IMAGE"),
        'IMAGE_VERSION':np.string_("1.2")}
    #Write hdf5 file from file_dict 
    with hiisi.HiisiHDF(output_h5, 'w') as h:
        h.create_from_filedict(file_dict_accum)


        

def main():

    config_file = f'config_dbzhtorate_{options.config}.json'
    coef, input_conf, output_conf = read_config(config_file)
    
    # Read first file and convert to rate to initiate sum array and output file_dict
    first_timestep = input_conf['timeres']
    first_file = input_conf['dir'] + '/' + input_conf['filename'].format(timestamp=options.timestamp, fc_timestep=f'{first_timestep:03}', config=options.config)
    first_image_array, quantity, first_timestamp, gain, offset, nodata, undetect = read_hdf5(first_file)
    nodata_mask_first = (first_image_array == nodata)
    undetect_mask_first = (first_image_array == undetect)
    
    # Calculate look up table (lut) for dBZ -> rate conversion.
    lut = dbzh_to_rate.calc_lookuptable_dBZtoRR(input_conf['timeres'], coef, nodata, undetect, gain, offset)

    # Init arrays
    file_dict_accum = init_filedict_accumulation(first_file)
    first_image_array = dbzh_to_rate.dBZtoRR_lut(np.int_(first_image_array),lut)
    
    startdate_first = first_timestamp[0:8]
    starttime_first = first_timestamp[8:14]
    
    if output_conf["write_acrr_from_start"]:
        acc_rate_from_start = first_image_array / (60 * 60 / (input_conf['timeres'] * 60))
        acc_rate_from_start[nodata_mask_first] = np.nan
        acc_rate_from_start[undetect_mask_first] = 0
    if output_conf["write_acrr_fixed_step"]:
        acc_rate_fixed_timestep = first_image_array / (60 * 60 / (input_conf['timeres'] * 60))
        acc_rate_fixed_timestep[nodata_mask_first] = np.nan
        acc_rate_fixed_timestep[undetect_mask_first] = 0
        n_fixed = 0
        
        
        
    # Loop through forecast fields (excluding the first one)
    for timestep in range(input_conf['timeres'] * 2, input_conf['fc_len']+1, input_conf['timeres']):

        print('timestep:', timestep)

        # Input file: {timestamp}+{timestep}min_radar.fmippn.det_conf={config}.h5
        input_file = input_conf['dir'] + '/' + input_conf['filename'].format(timestamp=options.timestamp, fc_timestep=f'{timestep:03}', config=options.config)
        
        # Read image array hdf5's and convert dbzh to rain rate (mm/timeresolution)
        image_array, quantity, timestamp, gain, offset, nodata, undetect = read_hdf5(input_file)
        
        # Convert to precipitation rate and mask nodata = np.nan and undetect = 0 for sum calculation
        nodata_mask = (image_array == nodata)
        undetect_mask = (image_array == undetect)
        image_array = dbzh_to_rate.dBZtoRR_lut(np.int_(image_array),lut)
        image_array[nodata_mask] = np.nan
        image_array[undetect_mask] = 0

        
        if output_conf["write_acrr_from_start"]:
            
            # Calculate and write to file accumulated rate from beginning of forecast
            #acc_rate_from_start = acc_rate_from_start + image_array
            acc_rate_from_start = np.where(np.isnan(acc_rate_from_start), image_array, acc_rate_from_start + np.nan_to_num(image_array))
            
            if timestep % output_conf["timestep"] == 0:

                # Change nodata and undetect values from undetect = 0 and nodata = np.nan to undetect = -0.0001 and nodata = 0 before
                # writing to file

                #print('acc_rate_from_start:', acc_rate_from_start)
                
                nodata_mask = ~np.isfinite(acc_rate_from_start) #(acc_rate_from_start == np.nan)
                undetect_mask = (acc_rate_from_start == 0)
                write_acc_rate_from_start = acc_rate_from_start
                write_acc_rate_from_start[undetect_mask] = -0.0001
                write_acc_rate_from_start[nodata_mask] = 0
                
                outfile = output_conf['dir'] + '/' + output_conf['filename'].format(timestamp=options.timestamp, fc_timestep=f'{timestep:03}', acc_timestep=f'{timestep:03}', config=options.config)
                
                startdate = startdate_first
                starttime = starttime_first
                enddate = timestamp[0:8]
                endtime = timestamp[8:14]
                date = enddate
                time = endtime
                
                write_accumulated_h5(outfile, write_acc_rate_from_start, file_dict_accum, date, time, startdate, starttime, enddate, endtime)


        
        if output_conf["write_acrr_fixed_step"]:

            #acc_rate_fixed_timestep = acc_rate_fixed_timestep + image_array
            acc_rate_fixed_timestep = np.where(np.isnan(acc_rate_fixed_timestep), image_array, acc_rate_fixed_timestep + np.nan_to_num(image_array))

            
            if n_fixed == 0:
                startdate = timestamp[0:8]
                starttime = timestamp[8:14]

            n_fixed += 1
            
            if timestep % output_conf["timestep"] == 0:

                nodata_mask = ~np.isfinite(acc_rate_fixed_timestep)
                undetect_mask = (acc_rate_fixed_timestep == 0)
                write_acc_rate_fixed_timestep = acc_rate_fixed_timestep
                write_acc_rate_fixed_timestep[undetect_mask] = -0.0001
                write_acc_rate_fixed_timestep[nodata_mask] = 0

                
                #Write to file
                outfile = output_conf['dir'] + '/' + output_conf['filename'].format(timestamp=options.timestamp, fc_timestep=f'{timestep:03}', acc_timestep=f'{output_conf["timestep"]:03}', config=options.config)
                
                enddate = timestamp[0:8]
                endtime = timestamp[8:14]
                date = enddate
                time = endtime
                
                write_accumulated_h5(outfile, write_acc_rate_fixed_timestep, file_dict_accum, date, time, startdate, starttime, enddate, endtime)

                #Init next sum array
                acc_rate_fixed_timestep = np.zeros_like(acc_rate_fixed_timestep)
                n_fixed = 0

                
if __name__ == '__main__':
    #Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--timestamp',
                        type = str,
                        default = '202201170700',
                        help = 'Input timestamp')
    parser.add_argument('--config',
                        type = str,
                        default = 'ravake',
                        help = 'Config file to use.')

    options = parser.parse_args()
    main()
