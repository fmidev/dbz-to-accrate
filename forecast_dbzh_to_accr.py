import numpy as np
import argparse
import datetime

import dbzh_to_rate
import utils


def run(timestamp, config):

    config_file = f'/config/{config}.json'
    coef, interp_conf, snowprob_conf, input_conf, output_conf = utils.read_config(config_file)
    timestamp_formatted = datetime.datetime.strptime(timestamp, "%Y%m%d%H%M")

    # Read first file and convert to rate to initiate sum array and output file_dict
    first_timestep = input_conf['timeres']
    second_timestamp = (timestamp_formatted + datetime.timedelta(minutes=first_timestep)).strftime('%Y%m%d%H%M%S')
    first_file = input_conf['dir'] + '/' + input_conf['filename'].format(timestamp=f'{timestamp}00', fc_timestamp=second_timestamp, config=config)
    first_image_array, quantity, first_timestamp, gain, offset, nodata, undetect = utils.read_hdf5(first_file)
    print("gain, offset, nodata, undetect ", gain, offset, nodata, undetect)
    nodata_mask_first = (first_image_array == nodata)
    undetect_mask_first = (first_image_array == undetect)

    # Read probability of snow from file
    snowprob_file = snowprob_conf['dir'] + '/' + snowprob_conf['filename'].format(timestamp=timestamp_formatted.strftime("%Y%m%d%H%M"))
    snowprob, snowprob_quantity, snowprob_timestamp, snowprob_gain, snowprob_offset, snowprob_nodata, snowprob_undetect = utils.read_hdf5(snowprob_file)

    # Calculate look up table (lut) for dBZ -> rate conversion.
    lut_rr, lut_sr = dbzh_to_rate.calc_lookuptables_dBZtoRATE(input_conf['timeres'], coef, nodata, undetect, gain, offset)

    print("np.min(lut_rr), np.min(lut_sr)", np.min(lut_rr), np.min(lut_sr))

    # Init arrays
    file_dict_accum = utils.init_filedict_accumulation(first_file)
    first_image_array = dbzh_to_rate.dBZtoRATE_lut(np.int_(first_image_array), lut_rr, lut_sr, snowprob)

    startdate_first = first_timestamp[0:8]
    starttime_first = first_timestamp[8:14]

    if output_conf["write_acrr_fixed_step"]:
        acc_rate_fixed_timestep = np.full_like(first_image_array, np.nan)
        n_fixed = 0
    if output_conf["write_acrr_fixed_step_2"]:
        acc_rate_fixed_timestep_2 = np.full_like(first_image_array, np.nan)
        n_fixed_2 = 0
    if output_conf["write_acrr_from_start"]:
        acc_rate_from_start = np.full_like(first_image_array, np.nan)


    # Loop through forecast fields
    for timestep in range(input_conf['timeres'], input_conf['fc_len']+1, input_conf['timeres']):

        # Input file
        second_timestamp = (timestamp_formatted + datetime.timedelta(minutes=timestep)).strftime('%Y%m%d%H%M%S')
        input_file = input_conf['dir'] + '/' + input_conf['filename'].format(timestamp = timestamp, fc_timestamp = second_timestamp, config=config)

        # Read image array hdf5's and convert dbzh to rain rate (mm/timeresolution)
        image_array, quantity, fc_timestamp, gain, offset, nodata, undetect = utils.read_hdf5(input_file)

        # Convert to precipitation rate and mask nodata = np.nan and undetect = 0 for sum calculation
        nodata_mask = (image_array == nodata)
        undetect_mask = (image_array == undetect)
        image_array = dbzh_to_rate.dBZtoRATE_lut(np.int_(image_array), lut_rr, lut_sr, snowprob)
        image_array[nodata_mask] = np.nan
        image_array[undetect_mask] = 0

        # Calculate sum and write to file in fixed time interval
        if output_conf["write_acrr_fixed_step"]:

            acc_rate_fixed_timestep = np.where(np.isnan(acc_rate_fixed_timestep), image_array, acc_rate_fixed_timestep + np.nan_to_num(image_array))

            if n_fixed == 0:
                startdate = fc_timestamp[0:8]
                starttime = fc_timestamp[8:14]

            n_fixed += 1

            if timestep % output_conf["timestep"] == 0:

                nodata_mask = ~np.isfinite(acc_rate_fixed_timestep)
                undetect_mask = (acc_rate_fixed_timestep == 0)
                write_acc_rate_fixed_timestep = acc_rate_fixed_timestep

                print("Before dtype conversion: np.nanmin(write_acc_rate_fixed_timestep): ", np.nanmin(write_acc_rate_fixed_timestep))
                print("Before dtype conversion: np.nanmax(write_acc_rate_fixed_timestep): ", np.nanmax(write_acc_rate_fixed_timestep))

                write_acc_rate_fixed_timestep = utils.convert_dtype(write_acc_rate_fixed_timestep, output_conf, nodata_mask, undetect_mask)

                print("After dtype conversion: np.min(write_acc_rate_fixed_timestep): ", np.min(write_acc_rate_fixed_timestep))
                print("After dtype conversion: np.max(write_acc_rate_fixed_timestep[(write_acc_rate_fixed_timestep > 0) & (write_acc_rate_fixed_timestep < 65535)]): ", np.max(write_acc_rate_fixed_timestep[(write_acc_rate_fixed_timestep > 0) & (write_acc_rate_fixed_timestep < 65535)]))

                #Write to file
                outfile = output_conf['dir'] + '/' + output_conf['filename'].format(timestamp = timestamp, fc_timestamp = second_timestamp, acc_timestep = f'{output_conf["timestep"]:03}', config = config)
                enddate = fc_timestamp[0:8]
                endtime = fc_timestamp[8:14]
                date = enddate
                time = endtime

                utils.write_accumulated_h5(outfile, write_acc_rate_fixed_timestep, file_dict_accum, date, time, startdate, starttime, enddate, endtime, output_conf)

                #Init next sum array
                acc_rate_fixed_timestep = np.full_like(acc_rate_fixed_timestep, np.nan)
                n_fixed = 0


        # Calculate sum and write to file in fixed time interval
        if output_conf["write_acrr_fixed_step_2"]:

            acc_rate_fixed_timestep_2 = np.where(np.isnan(acc_rate_fixed_timestep_2), image_array, acc_rate_fixed_timestep_2 + np.nan_to_num(image_array))

            if n_fixed_2 == 0:
                startdate = fc_timestamp[0:8]
                starttime = fc_timestamp[8:14]

            n_fixed_2 += 1

            if timestep % output_conf["timestep_2"] == 0:

                nodata_mask = ~np.isfinite(acc_rate_fixed_timestep_2)
                undetect_mask = (acc_rate_fixed_timestep_2 == 0)
                write_acc_rate_fixed_timestep_2 = acc_rate_fixed_timestep_2

                write_acc_rate_fixed_timestep_2 = utils.convert_dtype(write_acc_rate_fixed_timestep_2, output_conf, nodata_mask, undetect_mask)

                #Write to file
                outfile = output_conf['dir'] + '/' + output_conf['filename'].format(timestamp = timestamp, fc_timestamp = second_timestamp, acc_timestep = f'{output_conf["timestep_2"]:03}', config = config)
                enddate = fc_timestamp[0:8]
                endtime = fc_timestamp[8:14]
                date = enddate
                time = endtime

                utils.write_accumulated_h5(outfile, write_acc_rate_fixed_timestep_2, file_dict_accum, date, time, startdate, starttime, enddate, endtime, output_conf)

                #Init next sum array
                acc_rate_fixed_timestep_2 = np.full_like(acc_rate_fixed_timestep_2, np.nan)
                n_fixed_2 = 0


        # Calculate and write to file accumulated rate from beginning of forecast
        if output_conf["write_acrr_from_start"]:

            acc_rate_from_start = np.where(np.isnan(acc_rate_from_start), image_array, acc_rate_from_start + np.nan_to_num(image_array))

            if timestep % output_conf["timestep"] == 0:

                # Change nodata and undetect values from undetect = 0 and nodata = np.nan to undetect = -0.0001 and nodata = 0 before
                # writing to file
                nodata_mask = ~np.isfinite(acc_rate_from_start) #(acc_rate_from_start == np.nan)
                undetect_mask = (acc_rate_from_start == 0)
                write_acc_rate_from_start = acc_rate_from_start

                write_acc_rate_from_start = utils.convert_dtype(write_acc_rate_from_start, output_conf, nodata_mask, undetect_mask)

                outfile = output_conf['dir'] + '/' + output_conf['filename'].format(timestamp = timestamp, fc_timestamp = second_timestamp, acc_timestep = f'{timestep:03}', config=config)

                startdate = startdate_first
                starttime = starttime_first
                enddate = fc_timestamp[0:8]
                endtime = fc_timestamp[8:14]
                date = enddate
                time = endtime

                utils.write_accumulated_h5(outfile, write_acc_rate_from_start, file_dict_accum, date, time, startdate, starttime, enddate, endtime, output_conf)


def main():

    run(options.timestamp, options.config)


if __name__ == '__main__':
    #Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--timestamp',
                        type = str,
                        default = '2022011707',
                        help = 'Input timestamp')
    parser.add_argument('--config',
                        type = str,
                        default = 'ravake',
                        help = 'Config file to use.')

    options = parser.parse_args()
    main()
