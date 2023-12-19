"""Calculate the rainfall accumulation from the interpolated ensemble data."""
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sys

import utils


def load_file(ensno, timestep, curdate, conf, config, file_dict_accum=None):
    infile = Path(
        conf["interpolation_output"]["dir"]
        + "/"
        + conf["interpolation_output"]["filename"].format(
            timestamp=f"{curdate:%Y%m%d%H%M}00",
            fc_timestep=f"{(timestep - curdate).total_seconds() / 60:.0f}",
            fc_timestamp=f"{timestep:%Y%m%d%H%M}00",
            ensno=ensno,
            config=config,
        )
    )
    try:
        arr, qty, tstamp, gain, offset, nodata, undetect = utils.read_hdf5(infile, qty="ACRR")
    except FileNotFoundError:
        logging.warning(f"File {infile} not found, skipping")
        return None, None, None, None

    # Unpack dtype
    arr, nodata_mask, undetect_mask = utils.unpack_dtype(arr, conf["interpolation_output"])

    if file_dict_accum is None:
        file_dict_accum = utils.init_filedict_accumulation(infile)

    return arr, nodata_mask, undetect_mask, file_dict_accum


def run(timestamp, config):
    """
    Run the accumulation calculation process.

    TODO:
    - take into account the snow probability when converting dBZ to rain rate

    Args:
        timestamp (str): The timestamp in the format "%Y%m%d%H%M".
        config (str): The name of the configuration file.

    Returns:
        None

    """
    # Read config file
    config_file = f"/config/{config}.json"
    conf = utils.read_conf(config_file)
    curdate = datetime.strptime(timestamp, "%Y%m%d%H%M")

    timestep = conf["ensemble_input"]["timeres"]

    accumulation_times = utils.determine_accumulation_times(curdate, conf)
    accumulation_timestep = conf["output"]["timestep"]

    FILE_DICT_ACCUM = None
    # Calculate accumulations for the times that are needed
    for end in accumulation_times:
        start = end - timedelta(minutes=accumulation_timestep)

        acrr_timesteps = pd.date_range(
            start=start,
            end=end,
            freq=f"{timestep}min",
            inclusive="right",
        ).to_pydatetime()

        accrs = {}
        nodata_masks = {}
        undetect_masks = {}
        logging.info(f"Processing accumulation timestep {start}")

        missing_ensemble_members = []
        for ensno_ in range(1, conf["ensemble_input"]["n_ens_members"] + 1):
            interp_arrs = {}
            for tt in acrr_timesteps:
                # Read ensemble member
                # import pdb

                # pdb.set_trace()
                arr, _, _, file_dict_accum = load_file(
                    ensno_,
                    tt,
                    curdate,
                    conf,
                    config,
                    file_dict_accum=None,
                )
                if arr is not None:
                    interp_arrs[tt] = arr
                if FILE_DICT_ACCUM is None:
                    FILE_DICT_ACCUM = file_dict_accum

            # Sum 5min accumulations to accumulations of required length
            logging.debug(f"Processing ensemble member {ensno_}/{conf['ensemble_input']['n_ens_members']}")
            # Get keys that are in the interval
            # TODO if this is too low, we should reject the accumulation as invalid
            keys_in_interval = [k for k in interp_arrs.keys() if k > start and k <= end]

            logging.debug(f"No of keys in interval: {len(keys_in_interval)}")
            if len(keys_in_interval) < accumulation_timestep / timestep:
                missing_ensemble_members.append(ensno_)

                if len(missing_ensemble_members) > conf["ensemble_input"]["allow_n_members_missing"]:
                    logging.warning(f"Too many ensemble members missing, stopping accumulation calculation")
                    sys.exit(1)

                logging.warning(
                    f"Skipping ensemble member {ensno_}, only {len(keys_in_interval)} timesteps in interval"
                )
                continue

            # Get accumulation
            arrs = [interp_arrs[k] for k in keys_in_interval]
            accrs[ensno_] = np.nansum(arrs, axis=0)

            # Get nodata and undetect masks
            nodata_arrs = [np.isnan(interp_arrs[k]) for k in keys_in_interval]
            undetect_arrs = [interp_arrs[k] == 0 for k in keys_in_interval]
            # TODO figure out if here should be any() or all()
            nodata_masks[ensno_] = np.all(nodata_arrs, axis=0)
            undetect_masks[ensno_] = np.all(undetect_arrs, axis=0)

        # Ensemble mean for the accumulation
        ens_mean = np.nanmean([accrs[k] for k in accrs.keys()], axis=0)
        ens_nodata_mask = np.all([nodata_masks[k] for k in accrs.keys()], axis=0)
        ens_undetect_mask = np.all([undetect_masks[k] for k in accrs.keys()], axis=0)
        ens_mean_ = utils.convert_dtype(
            ens_mean,
            conf["output"],
            ens_nodata_mask,
            ens_undetect_mask,
        )

        # Write accumulation to file
        outfile = Path(
            conf["output"]["dir"]
            + "/"
            + conf["output"]["filename"].format(
                timestamp=f"{timestamp}00",
                fc_timestep=f"{(end - curdate).total_seconds() / 60:.0f}",
                fc_timestamp=f"{end:%Y%m%d%H%M}00",
                acc_timestep=f'{conf["output"]["timestep"]:03}',
                config=config,
            )
        )
        enddate = f"{end:%Y%m%d}"
        endtime = f"{end:%H%M}"
        startdate = f"{start:%Y%m%d}"
        starttime = f"{start:%H%M}"

        utils.write_accumulated_h5(
            outfile,
            ens_mean_,
            FILE_DICT_ACCUM,
            enddate,
            endtime,
            startdate,
            starttime,
            enddate,
            endtime,
            conf["output"],
        )

        # Write nodata and undetect masks to file
        nodata_outfile = outfile.parent / (outfile.stem + "_nodata_mask.h5")
        ens_nodata_mask_ = utils.convert_dtype(
            ens_nodata_mask,
            conf["nodata_output"],
            ens_nodata_mask,
            ens_undetect_mask,
        )

        utils.write_accumulated_h5(
            nodata_outfile,
            ens_nodata_mask_,
            FILE_DICT_ACCUM,
            enddate,
            endtime,
            startdate,
            starttime,
            enddate,
            endtime,
            conf["nodata_output"],
            quantity="nodata_mask",
        )

        undetect_outfile = outfile.parent / (outfile.stem + "_undetect_mask.h5")
        ens_undetect_mask_ = utils.convert_dtype(
            ens_undetect_mask,
            conf["undetect_output"],
            ens_nodata_mask,
            ens_undetect_mask,
        )
        utils.write_accumulated_h5(
            undetect_outfile,
            ens_undetect_mask_,
            FILE_DICT_ACCUM,
            enddate,
            endtime,
            startdate,
            starttime,
            enddate,
            endtime,
            conf["undetect_output"],
            quantity="undetect_mask",
        )


if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timestamp", type=str, default="202310300845", help="Input timestamp")
    parser.add_argument("--config", type=str, default="ravake-ens", help="Config file to use.")

    options = parser.parse_args()

    # Set logging level
    logging.basicConfig(level=logging.INFO)

    run(options.timestamp, options.config)
