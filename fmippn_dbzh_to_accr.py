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
import utils


def main():

    config_file = f'/config/{options.config}.json'
    coef, interp_conf, input_conf, output_conf = utils.read_config(config_file)
    
    # Read first file and convert to rate to initiate sum array and output file_dict
    first_timestep = input_conf['timeres']
    first_file = input_conf['dir'] + '/' + input_conf['filename'].format(timestamp=options.timestamp, fc_timestep=f'{first_timestep:03}', config=options.config)
    first_image_array, quantity, first_timestamp, gain, offset, nodata, undetect = utils.read_hdf5(first_file)
    nodata_mask_first = (first_image_array == nodata)
    undetect_mask_first = (first_image_array == undetect)
    
    # Calculate look up table (lut) for dBZ -> rate conversion.
    lut = dbzh_to_rate.calc_lookuptable_dBZtoRR(input_conf['timeres'], coef, nodata, undetect, gain, offset)

    # Init arrays
    file_dict_accum = utils.init_filedict_accumulation(first_file)
    first_image_array = dbzh_to_rate.dBZtoRR_lut(np.int_(first_image_array),lut)
    
    startdate_first = first_timestamp[0:8]
    starttime_first = first_timestamp[8:14]

    if output_conf["write_acrr_fixed_step"]:
        acc_rate_fixed_timestep = np.full_like(first_image_array, np.nan)
        n_fixed = 0    
    if output_conf["write_acrr_from_start"]:
        acc_rate_from_start = np.full_like(first_image_array, np.nan)
        
    # Loop through forecast fields
    for timestep in range(input_conf['timeres'], input_conf['fc_len']+1, input_conf['timeres']):

        print('timestep:', timestep)

        # Input file: {timestamp}+{timestep}min_radar.fmippn.det_conf={config}.h5
        input_file = input_conf['dir'] + '/' + input_conf['filename'].format(timestamp=options.timestamp, fc_timestep=f'{timestep:03}', config=options.config)
        
        # Read image array hdf5's and convert dbzh to rain rate (mm/timeresolution)
        image_array, quantity, timestamp, gain, offset, nodata, undetect = utils.read_hdf5(input_file)
        
        # Convert to precipitation rate and mask nodata = np.nan and undetect = 0 for sum calculation
        nodata_mask = (image_array == nodata)
        undetect_mask = (image_array == undetect)
        image_array = dbzh_to_rate.dBZtoRR_lut(np.int_(image_array),lut)
        image_array[nodata_mask] = np.nan
        image_array[undetect_mask] = 0

        
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
                
                write_acc_rate_fixed_timestep = utils.convert_dtype(write_acc_rate_fixed_timestep, output_conf, nodata_mask, undetect_mask)

                #Write to file
                outfile = output_conf['dir'] + '/' + output_conf['filename'].format(timestamp=options.timestamp, fc_timestep=f'{timestep:03}', acc_timestep=f'{output_conf["timestep"]:03}', config=options.config)
                enddate = timestamp[0:8]
                endtime = timestamp[8:14]
                date = enddate
                time = endtime
                
                utils.write_accumulated_h5(outfile, write_acc_rate_fixed_timestep, file_dict_accum, date, time, startdate, starttime, enddate, endtime, output_conf)

                #Init next sum array
                acc_rate_fixed_timestep = np.full_like(acc_rate_fixed_timestep, np.nan)
                n_fixed = 0

                
        if output_conf["write_acrr_from_start"]:
            
            # Calculate and write to file accumulated rate from beginning of forecast
            acc_rate_from_start = np.where(np.isnan(acc_rate_from_start), image_array, acc_rate_from_start + np.nan_to_num(image_array))
            
            if timestep % output_conf["timestep"] == 0:

                # Change nodata and undetect values from undetect = 0 and nodata = np.nan to undetect = -0.0001 and nodata = 0 before
                # writing to file                
                nodata_mask = ~np.isfinite(acc_rate_from_start) #(acc_rate_from_start == np.nan)
                undetect_mask = (acc_rate_from_start == 0)
                write_acc_rate_from_start = acc_rate_from_start
                
                write_acc_rate_from_start = utils.convert_dtype(write_acc_rate_from_start, output_conf, nodata_mask, undetect_mask)
                
                outfile = output_conf['dir'] + '/' + output_conf['filename'].format(timestamp=options.timestamp, fc_timestep=f'{timestep:03}', acc_timestep=f'{timestep:03}', config=options.config)
                
                startdate = startdate_first
                starttime = starttime_first
                enddate = timestamp[0:8]
                endtime = timestamp[8:14]
                date = enddate
                time = endtime
                
                utils.write_accumulated_h5(outfile, write_acc_rate_from_start, file_dict_accum, date, time, startdate, starttime, enddate, endtime, output_conf)
                
                
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
