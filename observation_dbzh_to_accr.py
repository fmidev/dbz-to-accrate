import numpy as np
import argparse
import datetime
from pathlib import Path

import dbzh_to_rate
import utils
import advection_correction


def run(timestamp, config, use_snowprob=True):
    config_file = f"/config/{config}.json"
    coef, interp_conf, snowprob_conf, input_conf, output_conf = utils.read_config(config_file)

    # Get current and earlier timestamp
    second_timestamp = timestamp
    second_timestep = datetime.datetime.strptime(second_timestamp, "%Y%m%d%H%M")
    first_timestamp = (second_timestep - datetime.timedelta(minutes=(input_conf["timeres"]))).strftime("%Y%m%d%H%M")
    first_timestep = second_timestep - datetime.timedelta(minutes=(input_conf["timeres"]))

    # Read image array hdf5's
    first_file = Path(
        input_conf["dir"].format(
            year=first_timestep.strftime("%Y"),
            month=first_timestep.strftime("%m"),
            day=first_timestep.strftime("%d"),
        )
    ) / input_conf["filename"].format(timestamp=first_timestamp)
    second_file = Path(
        input_conf["dir"].format(
            year=second_timestep.strftime("%Y"),
            month=second_timestep.strftime("%m"),
            day=second_timestep.strftime("%d"),
        )
    ) / input_conf["filename"].format(timestamp=second_timestamp)

    (
        first_image_array,
        quantity,
        first_timestamp_odim,
        gain,
        offset,
        nodata,
        undetect,
    ) = utils.read_hdf5(first_file, qty="DBZH")
    (
        second_image_array,
        quantity,
        second_timestamp_odim,
        gain,
        offset,
        nodata,
        undetect,
    ) = utils.read_hdf5(second_file, qty="DBZH")
    nodata_mask_first = first_image_array == nodata
    undetect_mask_first = first_image_array == undetect
    nodata_mask_second = second_image_array == nodata
    undetect_mask_second = second_image_array == undetect

    # Calculate look up tables (lut) for dBZ -> rate conversion.
    lut_rr, lut_sr = dbzh_to_rate.calc_lookuptables_dBZtoRATE(
        interp_conf["timeres"], coef, nodata, undetect, gain, offset
    )

    # Read probability of snow in array from file. Use snow probability
    # file of first timestamp to avoid having to wait for newer data.
    if use_snowprob:
        snowprob = utils.read_snowprob(first_timestep, snowprob_conf)
        snow_threshold = snowprob_conf.get("snow_threshold")

        # Convert image arrays dBZ -> rate
        first_image_array = dbzh_to_rate.dBZtoRATE_lut(
            np.int_(first_image_array), lut_rr, lut_sr, snowprob, snow_threshold=snow_threshold
        )
        second_image_array = dbzh_to_rate.dBZtoRATE_lut(
            np.int_(second_image_array), lut_rr, lut_sr, snowprob, snow_threshold=snow_threshold
        )
    else:
        first_image_array = dbzh_to_rate.dBZtoRR_lut(np.int_(first_image_array), lut_rr)
        second_image_array = dbzh_to_rate.dBZtoRR_lut(np.int_(second_image_array), lut_rr)

    # Init output file_dict
    file_dict_accum = utils.init_filedict_accumulation(first_file)

    # Change nodata and undetect to zero and np.nan before interpolation
    first_image_array[nodata_mask_first] = np.nan
    first_image_array[undetect_mask_first] = 0
    second_image_array[nodata_mask_second] = np.nan
    second_image_array[undetect_mask_second] = 0

    # Call interpolation
    R = np.array([first_image_array, second_image_array])
    R_interp = advection_correction.advection_correction(R, input_conf["timeres"], interp_conf["timeres"])

    # Init sum array and calculate sum
    acc_rate = np.full_like(first_image_array, np.nan)
    for i in range(0, len(R_interp)):
        acc_rate = np.where(np.isnan(acc_rate), R_interp[i], acc_rate + np.nan_to_num(R_interp[i]))

    nodata_mask = ~np.isfinite(acc_rate)
    undetect_mask = acc_rate == 0
    acc_rate = utils.convert_dtype(acc_rate, output_conf["accrate"], nodata_mask, undetect_mask)

    # Write to file
    outdir = Path(
        output_conf["accrate"]["dir"].format(
            year=second_timestamp[0:4],
            month=second_timestamp[4:6],
            day=second_timestamp[6:8],
        )
    )
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / output_conf["accrate"]["filename"].format(
        timestamp=timestamp, timeres=f'{input_conf["timeres"]:03}', config=config
    )

    startdate = f"{first_timestep:%Y%m%d}"
    starttime = f"{first_timestep:%H%M00}"
    enddate = f"{second_timestep:%Y%m%d}"
    endtime = f"{second_timestep:%H%M00}"
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
        output_conf["accrate"],
    )

    # Write rain rate to file
    nodata_mask = ~np.isfinite(second_image_array)
    undetect_mask = second_image_array == 0
    # Convert from mm/Tmin to mm/h
    second_image_array = second_image_array * (60 / interp_conf["timeres"])
    rate = utils.convert_dtype(second_image_array, output_conf["rate"], nodata_mask, undetect_mask)
    outdir = Path(
        output_conf["rate"]["dir"].format(
            year=second_timestamp[0:4],
            month=second_timestamp[4:6],
            day=second_timestamp[6:8],
        )
    )
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / output_conf["rate"]["filename"].format(
        timestamp=timestamp, timeres=f'{input_conf["timeres"]:03}', config=config
    )
    startdate = f"{second_timestep:%Y%m%d}"
    starttime = f"{second_timestep:%H%M00}"
    enddate = f"{second_timestep:%Y%m%d}"
    endtime = f"{second_timestep:%H%M00}"
    date = enddate
    time = endtime
    utils.write_accumulated_h5(
        outfile,
        rate,
        file_dict_accum,
        date,
        time,
        startdate,
        starttime,
        enddate,
        endtime,
        output_conf["rate"],
        quantity="RATE",
    )


if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, default="202201170700", help="Input timestamp")
    parser.add_argument("--config", type=str, default="ravake_composite", help="Config file to use.")
    parser.add_argument("--no-snowprob", action="store_false", dest="use_snowprob", help="Use snow probability")

    options = parser.parse_args()
    run(options.timestamp, options.config, options.use_snowprob)
