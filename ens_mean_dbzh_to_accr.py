"""Calculate ensemble mean accumulated rain rate."""
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging

from pysteps import motion

import dbzh_to_rate
import utils
import advection_correction


def load_file(file, timestep, conf, lut_rr=None, lut_sr=None, file_dict_accum=None):
    arr, qty, tstamp, gain, offset, nodata, undetect = utils.read_hdf5(file)
    # Convert to rain rate
    nodata_mask = arr == nodata
    undetect_mask = arr == undetect

    if lut_rr is None:
        lut_rr, lut_sr = dbzh_to_rate.calc_lookuptables_dBZtoRATE(
            timestep, conf["coef"], nodata, undetect, gain, offset
        )
        file_dict_accum = utils.init_filedict_accumulation(file)

    arr = dbzh_to_rate.dBZtoRR_lut(np.int_(arr), lut_rr)
    arr[nodata_mask] = np.nan
    arr[undetect_mask] = 0

    return arr, nodata_mask, undetect_mask, file_dict_accum, lut_rr, lut_sr


def interpolate_and_sum(arr1, arr2, motion_field, conf):
    # Calculate advection correction
    interp_frames = advection_correction.advection_correction_with_motion(
        np.array([arr1, arr2]),
        motion_field,
        T=conf["input"]["timeres"],
        t=conf["interp"]["timeres"],
    )

    # Calculate 5 min accumulated rain rate
    arr = np.nansum(interp_frames, axis=0)
    return arr


def run(timestamp, config, configpath="/config"):
    # TODO
    # - no data mask
    # - undetect mask
    # - lumitodennäköisyyden huomiointi
    # - kertymän laskenta tasatunneittain havainnot huomoiden

    # Read config file
    config_file = f"/config/{config}.json"
    # coef, interp_conf, snowprob_conf, input_conf, output_conf = utils.read_config(config_file)
    conf = utils.read_conf(config_file)
    curdate = datetime.strptime(timestamp, "%Y%m%d%H%M")

    # Read first file to initialize sum array and output file_dict
    timestep = conf["input"]["timeres"]
    input_path = Path(conf["observations"]["dir"])
    first_file = input_path / conf["observations"]["filename"].format(
        timestamp=f"{timestamp}",
    )
    first_image_array, quantity, first_timestamp, gain, offset, nodata, undetect = utils.read_hdf5(first_file)
    nodata_mask_first = first_image_array == nodata
    undetect_mask_first = first_image_array == undetect

    # Calculate look up table (lut) for dBZ -> rate conversion.
    lut_rr_obs, lut_sr_obs = dbzh_to_rate.calc_lookuptables_dBZtoRATE(
        timestep, conf["coef"], nodata, undetect, gain, offset
    )
    lut_rr_ens = None
    lut_sr_ens = None
    file_dict_accum = None
    first_arr = dbzh_to_rate.dBZtoRR_lut(np.int_(first_image_array), lut_rr_obs)

    # Placeholder for snow probability handling, once the data is available
    # snowprob_file = (
    #     f"{snowprob_conf['dir']}/{snowprob_conf['filename'].format(timestamp=curdate.strftime('%Y%m%d%H%M'))}"
    # )
    # (
    #     snowprob,
    #     snowprob_quantity,
    #     snowprob_timestamp,
    #     snowprob_gain,
    #     snowprob_offset,
    #     snowprob_nodata,
    #     snowprob_undetect,
    # ) = utils.read_hdf5(snowprob_file)

    leadtimes = pd.date_range(
        start=curdate,
        end=curdate + timedelta(minutes=conf["input"]["fc_len"]),
        freq=f"{timestep}min",
        inclusive="right",
    ).to_pydatetime()

    accumulation_timestep = conf["output"]["timestep"]
    accumulation_times = pd.date_range(
        start=pd.Timestamp(curdate).floor(f"{accumulation_timestep}min"),
        end=curdate + timedelta(minutes=conf["input"]["fc_len"]),
        freq=f"{accumulation_timestep}min",
        inclusive="right",
    ).to_pydatetime()

    data_arrays = {i: {curdate: first_arr} for i in range(1, conf["input"]["n_ens_members"] + 1)}
    nodata_masks = {i: {curdate: nodata_mask_first} for i in range(1, conf["input"]["n_ens_members"] + 1)}
    undetect_masks = {i: {curdate: undetect_mask_first} for i in range(1, conf["input"]["n_ens_members"] + 1)}
    interp_arrays = {i: {} for i in range(1, conf["input"]["n_ens_members"] + 1)}
    # Load observations that are needed to calculate accumulations
    # if any are needed
    start = accumulation_times[0] - timedelta(minutes=accumulation_timestep)
    if start < curdate:
        obstimes = pd.date_range(
            start=start,
            end=curdate,
            freq=f"{conf['observations']['timeres']}min",
            inclusive="both",
        ).to_pydatetime()

        for tt in obstimes:
            logging.info(f"Processing observation {tt}")
            file = Path(conf["observations"]["dir"]) / conf["observations"]["filename"].format(
                timestamp=f"{tt:%Y%m%d%H%M}",
            )
            arr, nodata_mask, undetect_mask, _, lut_rr_obs, lut_sr_obs = load_file(
                file, timestep, conf, lut_rr_ens, lut_sr_ens, file_dict_accum=None
            )
            for ensno in range(1, conf["input"]["n_ens_members"] + 1):
                data_arrays[ensno][tt] = arr
                nodata_masks[ensno][tt] = nodata_mask
                undetect_masks[ensno][tt] = undetect_mask

            # Interpolate intermediate timesteps
            if conf["interp"]["interpolate"] and tt > obstimes[0]:
                arr1 = data_arrays[1][tt - timedelta(minutes=timestep)]
                arr2 = data_arrays[1][tt]

                oflow_method = motion.get_method("LK")
                fd_kwargs = {"buffer_mask": 10}  # avoid edge effects
                # NOTE: this line takes ~30% of the total runtime
                motion_field = oflow_method(np.log(np.stack([arr1, arr2])), fd_kwargs=fd_kwargs)

                # Calculate advection correction
                arr = interpolate_and_sum(arr1, arr2, motion_field, conf)
            for ensno in range(1, conf["input"]["n_ens_members"] + 1):
                interp_arrays[ensno][tt] = arr

            try:
                del data_arrays[1][tt - timedelta(minutes=timestep)]
            except KeyError:
                pass

    motion_fields = {}
    # For each ensemble member
    for ensno in range(1, conf["input"]["n_ens_members"] + 1):
        logging.info(f"Reading motion field for ensemble member {ensno}/{conf['input']['n_ens_members']}")
        # Read motion field for this ensemble member
        motion_file = Path(conf["motion"]["dir"]) / conf["motion"]["filename"].format(
            timestamp=f"{timestamp}00",
            ensno=ensno,
            config=config,
        )
        motion_field, motion_timestep, motion_kmperpixel = utils.read_motion_hdf5(motion_file)
        motion_fields[ensno] = utils._convert_motion_units_ms2pix(
            motion_field, kmperpixel=motion_timestep, timestep=motion_kmperpixel
        )

    # Calculate advection correction for each ensemble member
    for ensno in range(1, conf["input"]["n_ens_members"] + 1):
        logging.info(f"Processing ensemble member {ensno}/{conf['input']['n_ens_members']}")
        # lue datat täällä niin ei mene niin paljoa muistia
        for i, lt in enumerate(leadtimes):
            # Read file
            file = Path(conf["input"]["dir"]) / conf["input"]["filename"].format(
                timestamp=f"{timestamp}00",
                fc_timestamp=f"{lt:%Y%m%d%H%M}00",
                fc_timestep=f"{timestep:03}",
                config=config,
                ensno=ensno,
            )

            arr, nodata_mask, undetect_mask, file_dict_accum, lut_rr_ens, lut_sr_ens = load_file(
                file, timestep, conf, lut_rr_ens, lut_sr_ens, file_dict_accum
            )
            data_arrays[ensno][lt] = arr
            nodata_masks[ensno][lt] = nodata_mask
            undetect_masks[ensno][lt] = undetect_mask

        logging.info(f"Reading motion field for ensemble member {ensno}/{conf['input']['n_ens_members']}")
        # Read motion field for this ensemble member
        motion_file = Path(conf["motion"]["dir"]) / conf["motion"]["filename"].format(
            timestamp=f"{timestamp}00",
            ensno=ensno,
            config=config,
        )
        motion_field, motion_timestep, motion_kmperpixel = utils.read_motion_hdf5(motion_file)
        motion_fields[ensno] = utils._convert_motion_units_ms2pix(
            motion_field, kmperpixel=motion_timestep, timestep=motion_kmperpixel
        )

        logging.info(f"Calculating advection correction for ensemble member {ensno}/{conf['input']['n_ens_members']}")
        # Stack arrays to be interpolated so that we can calculate the interpolation only once
        adv_arrs_1 = np.stack([data_arrays[ensno][lt - timedelta(minutes=timestep)] for lt in leadtimes])
        adv_arrs_2 = np.stack([data_arrays[ensno][lt] for lt in leadtimes])

        # Do the interpolation
        # NOTE: Currently, this func takes ~50% of the total runtime
        R0 = advection_correction.interpolate_ensemble(adv_arrs_1, adv_arrs_2, motion_fields[ensno])

        # Extract interpolated frames
        interp_arrays[ensno] = {}
        for i, lt in enumerate(leadtimes):
            interp_arrays[ensno][lt] = R0[i]

        # Clear memory
        del adv_arrs_1
        del adv_arrs_2
        del R0
        del data_arrays[ensno]
        del motion_fields[ensno]

    # Calculate accumulations for the times that are needed
    for end in accumulation_times:
        start = end - timedelta(minutes=accumulation_timestep)
        accrs = {}
        logging.info(f"Processing accumulation timestep {start}")

        for ensno_ in range(1, conf["input"]["n_ens_members"] + 1):
            # Sum 5min accumulations to accumulations of required length
            logging.debug(f"Processing ensemble member {ensno_}/{conf['input']['n_ens_members']}")
            # Get keys that are in the interval
            # TODO if this is too low, we should reject the accumulation as invalid
            keys_in_interval = [k for k in interp_arrays[ensno_].keys() if k > start and k <= end]
            logging.debug(f"No of keys in interval: {len(keys_in_interval)}")
            arrs = [interp_arrays[ensno_][k] for k in keys_in_interval]
            accrs[ensno_] = np.nansum(arrs, axis=0)

        # Ensemble mean for the accumulation
        ens_mean = np.nanmean([accrs[k] for k in accrs.keys()], axis=0)
        ens_mean_ = utils.convert_dtype(
            ens_mean,
            conf["output"],
            nodata_mask,
            undetect_mask,
        )

        # Write accumulation to file
        outfile = (
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
            file_dict_accum,
            enddate,
            endtime,
            startdate,
            starttime,
            enddate,
            endtime,
            conf["output"],
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
