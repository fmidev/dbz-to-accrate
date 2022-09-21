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
from pathlib import Path

import dbzh_to_rate
import utils
import advection_correction


def main():

    config_file = f'/config/{options.config}.json'
    coef, interp_conf, input_conf, output_conf = utils.read_config(config_file)

    #Get current and earlier timestamp
    second_timestamp = options.timestamp
    formatted_time_second = datetime.datetime.strptime(second_timestamp,'%Y%m%d%H%M')
    first_timestamp = (formatted_time_second - datetime.timedelta(minutes=(input_conf['timeres']))).strftime('%Y%m%d%H%M')
    
    #Read image array hdf5's
    first_file = input_conf['dir'].format(year=first_timestamp[0:4], month=first_timestamp[4:6], day=first_timestamp[6:8]) + '/' + input_conf['filename'].format(timestamp=first_timestamp)
    second_file = input_conf['dir'].format(year=second_timestamp[0:4], month=second_timestamp[4:6], day=second_timestamp[6:8]) + '/' + input_conf['filename'].format(timestamp=second_timestamp)

    print('first_file: ', first_file)
    print('second_file: ', second_file)

    first_image_array, quantity, first_timestamp_odim, gain, offset, nodata, undetect = utils.read_hdf5(first_file)
    second_image_array, quantity, second_timestamp_odim, gain, offset, nodata, undetect = utils.read_hdf5(second_file)
    nodata_mask_first = (first_image_array == nodata)
    undetect_mask_first = (first_image_array == undetect)
    nodata_mask_second = (second_image_array == nodata)
    undetect_mask_second = (second_image_array == undetect)

    # Calculate look up table (lut) for dBZ -> rate conversion.
    lut = dbzh_to_rate.calc_lookuptable_dBZtoRR(interp_conf['timeres'], coef, nodata, undetect, gain, offset)

    # Init output file_dict 
    file_dict_accum = utils.init_filedict_accumulation(first_file)

    # Convert image arrays dBZ -> rate
    first_image_array = dbzh_to_rate.dBZtoRR_lut(np.int_(first_image_array),lut)
    second_image_array = dbzh_to_rate.dBZtoRR_lut(np.int_(second_image_array),lut)

    # Change nodata and undetect to zero and np.nan before interpolation
    first_image_array[nodata_mask_first] = np.nan
    first_image_array[undetect_mask_first] = 0
    second_image_array[nodata_mask_second] = np.nan
    second_image_array[undetect_mask_second] = 0

    # Call interpolation
    R = np.array([first_image_array, second_image_array])
    R_interp = advection_correction.advection_correction(R, input_conf['timeres'], interp_conf['timeres'])

    # Init sum array and calculate sum
    acc_rate = np.full_like(first_image_array, np.nan)
    for i in range(0,len(R_interp)):
        acc_rate = np.where(np.isnan(acc_rate), R_interp[i], acc_rate + np.nan_to_num(R_interp[i]))

        #Test plot
        plot_name='/output/pysteps_interp_' + str(i) + '.png'
        utils.plot_array(R_interp[i], plot_name)

    
    nodata_mask = ~np.isfinite(acc_rate)
    undetect_mask = (acc_rate == 0)
    acc_rate = utils.convert_dtype(acc_rate, output_conf, nodata_mask, undetect_mask)
    
    #Write to file
    outdir = output_conf['dir'].format(year=second_timestamp[0:4], month=second_timestamp[4:6], day=second_timestamp[6:8])
    print('outdir:', outdir)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    outfile = outdir + '/' + output_conf['filename'].format(timestamp=options.timestamp, timeres=input_conf['timeres'])
    startdate = first_timestamp[0:8]
    starttime = first_timestamp[8:14]
    enddate = second_timestamp[0:8] + "00"
    endtime = second_timestamp[8:14] + "00"
    date = enddate
    time = endtime
    print('date: ',enddate,' time: ',endtime)
    utils.write_accumulated_h5(outfile, acc_rate, file_dict_accum, date, time, startdate, starttime, enddate, endtime, output_conf)

    
                
if __name__ == '__main__':
    #Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--timestamp',
                        type = str,
                        default = '202201170700',
                        help = 'Input timestamp')
    parser.add_argument('--config',
                        type = str,
                        default = 'ravake_composite',
                        help = 'Config file to use.')

    options = parser.parse_args()
    main()
