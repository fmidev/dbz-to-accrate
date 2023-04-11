import numpy as np
import argparse
import datetime
from pathlib import Path

import dbzh_to_rate
import utils
import advection_correction


def run(timestamp, config):
    config_file = f"/config/{config}.json"
    coef, interp_conf, snowprob_conf, input_conf, output_conf = utils.read_config(
        config_file
    )

    # Get current and earlier timestamp
    second_timestamp = timestamp
    formatted_time_second = datetime.datetime.strptime(second_timestamp, "%Y%m%d%H%M")
    first_timestamp = (
        formatted_time_second - datetime.timedelta(minutes=(input_conf["timeres"]))
    ).strftime("%Y%m%d%H%M")

    # Read image array hdf5's
    if input_conf["dir_contains_date"]:
        first_file = f"{input_conf['dir'].format(year=first_timestamp[0:4], month=first_timestamp[4:6], day=first_timestamp[6:8])}/{input_conf['filename'].format(timestamp=first_timestamp)}"
        second_file = f"{input_conf['dir'].format(year=second_timestamp[0:4], month=second_timestamp[4:6], day=second_timestamp[6:8])}/{input_conf['filename'].format(timestamp=second_timestamp)}"
    else:
        first_file = f"{input_conf['dir']}/{input_conf['filename'].format(timestamp=first_timestamp)}"
        second_file = f"{input_conf['dir']}/{input_conf['filename'].format(timestamp=second_timestamp)}"

    (
        first_image_array,
        quantity,
        first_timestamp_odim,
        gain,
        offset,
        nodata,
        undetect,
    ) = utils.read_hdf5(first_file)
    (
        second_image_array,
        quantity,
        second_timestamp_odim,
        gain,
        offset,
        nodata,
        undetect,
    ) = utils.read_hdf5(second_file)
    nodata_mask_first = first_image_array == nodata
    undetect_mask_first = first_image_array == undetect
    nodata_mask_second = second_image_array == nodata
    undetect_mask_second = second_image_array == undetect

    # Read probability of snow in array from file
    snowprob_file = (
        snowprob_conf["dir"]
        + "/"
        + snowprob_conf["filename"].format(timestamp=timestamp)
    )
    (
        snowprob,
        snowprob_quantity,
        snowprob_timestamp,
        snowprob_gain,
        snowprob_offset,
        snowprob_nodata,
        snowprob_undetect,
    ) = utils.read_hdf5(snowprob_file)

    # Calculate look up tables (lut) for dBZ -> rate conversion.
    lut_rr, lut_sr = dbzh_to_rate.calc_lookuptables_dBZtoRATE(
        interp_conf["timeres"], coef, nodata, undetect, gain, offset
    )

    # Init output file_dict
    file_dict_accum = utils.init_filedict_accumulation(first_file)

    # Convert image arrays dBZ -> rate
    first_image_array = dbzh_to_rate.dBZtoRATE_lut(
        np.int_(first_image_array), lut_rr, lut_sr, snowprob
    )
    second_image_array = dbzh_to_rate.dBZtoRATE_lut(
        np.int_(second_image_array), lut_rr, lut_sr, snowprob
    )

    # Change nodata and undetect to zero and np.nan before interpolation
    first_image_array[nodata_mask_first] = np.nan
    first_image_array[undetect_mask_first] = 0
    second_image_array[nodata_mask_second] = np.nan
    second_image_array[undetect_mask_second] = 0

    # Call interpolation
    R = np.array([first_image_array, second_image_array])
    R_interp = advection_correction.advection_correction(
        R, input_conf["timeres"], interp_conf["timeres"]
    )

    # Init sum array and calculate sum
    acc_rate = np.full_like(first_image_array, np.nan)
    for i in range(0, len(R_interp)):
        acc_rate = np.where(
            np.isnan(acc_rate), R_interp[i], acc_rate + np.nan_to_num(R_interp[i])
        )

    nodata_mask = ~np.isfinite(acc_rate)
    undetect_mask = acc_rate == 0
    acc_rate = utils.convert_dtype(acc_rate, output_conf, nodata_mask, undetect_mask)

    # Write to file
    outdir = output_conf["dir"].format(
        year=second_timestamp[0:4],
        month=second_timestamp[4:6],
        day=second_timestamp[6:8],
    )
    Path(outdir).mkdir(parents=True, exist_ok=True)
    # outfile = f"{outdir}/{output_conf['filename'].format(timestamp=timestamp, timeres=input_conf['timeres'])}"
    outfile = f"""{outdir}/{output_conf['filename'].format(
        timestamp=timestamp,
        timeres=f'{input_conf["timeres"]:03}')}"""
    startdate = first_timestamp[0:8]
    starttime = first_timestamp[8:14]
    enddate = f"{second_timestamp[0:8]}00"
    endtime = f"{second_timestamp[8:14]}00"
    date = enddate
    time = endtime
    utils.write_accumulated_h5(
        outfile,
        acc_rate,
        file_dict_accum,
        date,
        time,
        startdate,
        starttime,
        enddate,
        endtime,
        output_conf,
    )


def main():
    run(options.timestamp, options.config)


if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timestamp", type=str, default="202201170700", help="Input timestamp"
    )
    parser.add_argument(
        "--config", type=str, default="ravake_composite", help="Config file to use."
    )

    options = parser.parse_args()
    main()
