"""Calculate the rainfall accumulation from the interpolated ensemble data."""
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sys

import utils


def load_ensemble_file(ensno, timestep, curdate, conf, config, file_dict_accum=None):
    infile = Path(
        conf["output"]["interpolation"]["dir"]
        + "/"
        + conf["output"]["interpolation"]["filename"].format(
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
    arr, nodata_mask, undetect_mask = utils.unpack_dtype(arr, conf["output"]["interpolation"])

    if file_dict_accum is None:
        file_dict_accum = utils.init_filedict_accumulation(infile)

    return arr, nodata_mask, undetect_mask, file_dict_accum


def load_det_file(timestep, curdate, conf, config, file_dict_accum=None):
    infile = Path(
        conf["input"]["deterministic"]["data"]["dir"]
        + "/"
        + conf["input"]["deterministic"]["data"]["filename"].format(
            timestamp=f"{curdate:%Y%m%d%H%M}00",
            fc_timestep=f"{(timestep - curdate).total_seconds() / 60:.0f}",
            fc_timestamp=f"{timestep:%Y%m%d%H%M}00",
            config=config,
        )
    )
    try:
        arr, qty, tstamp, gain, offset, nodata, undetect = utils.read_hdf5(infile, qty="DBZH")
    except FileNotFoundError:
        logging.warning(f"File {infile} not found, skipping")
        return None, None, None, None

    # Unpack dtype
    dtype_conf = {
        "gain": gain,
        "offset": offset,
        "nodata": nodata,
        "undetect": undetect,
    }
    arr, nodata_mask, undetect_mask = utils.unpack_dtype(arr, dtype_conf)

    if file_dict_accum is None:
        file_dict_accum = utils.init_filedict_accumulation(infile)

    return arr, nodata_mask, undetect_mask, file_dict_accum


def save_accr(
    arr,
    nodata_arr,
    undetect_arr,
    output_conf,
    starttime,
    endtime,
    timestamp,
    timestep,
    fc_timestep,
    fc_timestamp,
    config,
    FILE_DICT_ACCUM,
    quantity="ACRR",
):
    arr_ = utils.convert_dtype(
        arr,
        output_conf,
        nodata_arr,
        undetect_arr,
    )
    outfile = Path(output_conf["dir"]) / output_conf["filename"].format(
        timestamp=f"{timestamp}",
        fc_timestep=f"{fc_timestep:.0f}",
        fc_timestamp=f"{fc_timestamp:%Y%m%d%H%M}",
        acc_timestep=f"{timestep:03}",
        config=config,
    )
    utils.write_accumulated_h5(
        outfile,
        arr_,
        FILE_DICT_ACCUM,
        f"{endtime:%Y%m%d}",
        f"{endtime:%H%M}",
        f"{starttime:%Y%m%d}",
        f"{starttime:%H%M}",
        f"{endtime:%Y%m%d}",
        f"{endtime:%H%M}",
        output_conf,
        quantity=quantity,
    )


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

    timestep = conf["input"]["ensemble"]["data"]["timeres"]

    accumulation_times = utils.determine_accumulation_times(
        curdate, conf, fc_len=conf["output"]["accumulations"]["timeconfig"]["fc_len"]
    )
    accumulation_timestep = conf["output"]["accumulations"]["timeconfig"]["timestep"]

    ensemble_members = list(range(1, conf["input"]["ensemble"]["data"]["n_ens_members"] + 1))
    ensemble_members_det = ensemble_members + ["det"]

    # Weighting of deterministic and ensemble
    determ_initweight = conf["output"]["accumulations"]["weighted_mean_det_ens"]["determ_initweight"]
    determ_weightspan = conf["output"]["accumulations"]["weighted_mean_det_ens"]["determ_weightspan"]
    determ_startw = 0.01 * determ_initweight * conf["input"]["ensemble"]["data"]["n_ens_members"]
    n_timesteps = int(
        conf["output"]["accumulations"]["timeconfig"]["fc_len"]
        / conf["output"]["accumulations"]["timeconfig"]["timeres"]
    )
    determ_lapse = 100.0 * determ_startw / (determ_weightspan * n_timesteps)

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

        # Weights of deterministic forecast
        leadtime_indices = np.array([t.total_seconds() / 60 / timestep for t in acrr_timesteps - curdate])

        accrs = {}
        nodata_masks = {}
        undetect_masks = {}
        logging.info(f"Processing accumulation timestep {start} - {end}")

        missing_ensemble_members = []
        interp_arrs = {}
        for ensno_ in ensemble_members_det:
            interp_arrs[ensno_] = {}
            for tt in acrr_timesteps:
                # Read ensemble member
                arr, _, _, file_dict_accum = load_ensemble_file(
                    ensno_,
                    tt,
                    curdate,
                    conf,
                    config,
                    file_dict_accum=None,
                )
                if arr is not None:
                    interp_arrs[ensno_][tt] = arr
                if FILE_DICT_ACCUM is None:
                    FILE_DICT_ACCUM = file_dict_accum

            # Sum 5min accumulations to accumulations of required length
            logging.debug(f"Processing ensemble member {ensno_}/{conf['input']['ensemble']['data']['n_ens_members']}")
            # Get keys that are in the interval
            # TODO if this is too low, we should reject the accumulation as invalid
            keys_in_interval = [k for k in interp_arrs[ensno_].keys() if k > start and k <= end]

            logging.debug(f"No of keys in interval: {len(keys_in_interval)}")
            if len(keys_in_interval) < accumulation_timestep / timestep:
                missing_ensemble_members.append(ensno_)

                if len(missing_ensemble_members) > conf["input"]["ensemble"]["data"]["allow_n_members_missing"]:
                    logging.error(f"Too many ensemble members missing, stopping accumulation calculation")
                    sys.exit(1)

                logging.warning(
                    f"Skipping ensemble member {ensno_}, only {len(keys_in_interval)} timesteps in interval"
                )
                continue

            # Get accumulation
            arrs = [interp_arrs[ensno_][k] for k in keys_in_interval]
            accrs[ensno_] = np.nansum(arrs, axis=0)

            # Get nodata and undetect masks
            nodata_arrs = [np.isnan(interp_arrs[ensno_][k]) for k in keys_in_interval]
            undetect_arrs = [interp_arrs[ensno_][k] == 0 for k in keys_in_interval]
            # TODO figure out if here should be any() or all()
            nodata_masks[ensno_] = np.any(nodata_arrs, axis=0)
            undetect_masks[ensno_] = np.all(undetect_arrs, axis=0)

        existing_ensemble_members = list(set(ensemble_members) - set(missing_ensemble_members))

        # Ensemble mean for the accumulation
        ens_mean = np.nanmean([accrs[k] for k in existing_ensemble_members], axis=0)
        ens_nodata_mask = np.any([nodata_masks[k] for k in existing_ensemble_members], axis=0)
        ens_undetect_mask = np.all([undetect_masks[k] for k in existing_ensemble_members], axis=0)

        save_accr(
            ens_mean,
            ens_nodata_mask,
            ens_undetect_mask,
            conf["output"]["accumulations"]["ensemble_mean"],
            start,
            end,
            timestamp,
            conf["output"]["accumulations"]["timeconfig"]["timestep"],
            (end - curdate).total_seconds() / 60,
            end,
            config,
            FILE_DICT_ACCUM,
        )

        # Deterministic accumulation
        accr_arrays_det = np.stack([interp_arrs["det"][k] for i, k in enumerate(acrr_timesteps)])
        accr_weights_det = determ_startw - determ_lapse * leadtime_indices
        accr_det = np.nansum(accr_arrays_det, axis=0)
        det_nodata_mask = np.any([np.isnan(interp_arrs["det"][k]) for k in acrr_timesteps], axis=0)
        det_undetect_mask = np.all([interp_arrs["det"][k] == 0 for k in acrr_timesteps], axis=0)
        save_accr(
            accr_det,
            det_nodata_mask,
            det_undetect_mask,
            conf["output"]["accumulations"]["deterministic"],
            start,
            end,
            timestamp,
            conf["output"]["accumulations"]["timeconfig"]["timestep"],
            (end - curdate).total_seconds() / 60,
            end,
            config,
            FILE_DICT_ACCUM,
        )

        # Weighted average of deterministic and ensemble
        accr_arrays_ens = np.stack(
            [[interp_arrs[ensno_][k] for i, k in enumerate(acrr_timesteps)] for ensno_ in existing_ensemble_members]
        )
        accr_arrays = np.concatenate([accr_arrays_det[np.newaxis, :, :], accr_arrays_ens], axis=0)

        weights = np.concatenate(
            [accr_weights_det[np.newaxis, :], np.ones((len(existing_ensemble_members), len(acrr_timesteps)))], axis=0
        )

        # for loop version
        # ens_det_accr = np.zeros_like(ens_mean)
        # for t in range(len(acrr_timesteps)):
        #     avg_accr_t = np.average(accr_arrays[:, t, :, :], weights=weights[:, t], axis=0)
        #     ens_det_accr += avg_accr_t
        w1 = weights[:, :, None, None]
        scl = np.sum(w1, axis=0)
        avg1 = np.multiply(accr_arrays, w1).sum(axis=0) / scl

        # test equality with loop version
        # mask = ~(np.isnan(accr_weighted) | np.isnan(ens_det_accr))
        # np.allclose(accr_weighted[mask], ens_det_accr[mask])

        accr_weighted = np.sum(avg1, axis=0)
        # TODO check nodata masks
        weighted_nodata_mask = ens_nodata_mask | det_nodata_mask
        weighted_undetect_mask = ens_undetect_mask | det_undetect_mask
        save_accr(
            accr_weighted,
            weighted_nodata_mask,
            weighted_undetect_mask,
            conf["output"]["accumulations"]["weighted_mean_det_ens"],
            start,
            end,
            timestamp,
            conf["output"]["accumulations"]["timeconfig"]["timestep"],
            (end - curdate).total_seconds() / 60,
            end,
            config,
            FILE_DICT_ACCUM,
        )

        # Write nodata and undetect masks to file
        save_accr(
            ens_nodata_mask,
            ens_nodata_mask,
            ens_undetect_mask,
            conf["output"]["nodata"],
            start,
            end,
            timestamp,
            conf["output"]["accumulations"]["timeconfig"]["timestep"],
            (end - curdate).total_seconds() / 60,
            end,
            config,
            FILE_DICT_ACCUM,
            quantity="nodata_mask",
        )

        save_accr(
            ens_undetect_mask,
            ens_nodata_mask,
            ens_undetect_mask,
            conf["output"]["undetect"],
            start,
            end,
            timestamp,
            conf["output"]["accumulations"]["timeconfig"]["timestep"],
            (end - curdate).total_seconds() / 60,
            end,
            config,
            FILE_DICT_ACCUM,
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
