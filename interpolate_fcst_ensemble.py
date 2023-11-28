"""Interpolate forecast ensemble members to 1 min timestep."""
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging

import dask

from pysteps import motion

import dbzh_to_rate
import utils
import advection_correction


def load_file(file, timestep, conf, lut_rr=None, lut_sr=None, file_dict_accum=None):
    arr, qty, tstamp, gain, offset, nodata, undetect = utils.read_hdf5(file, qty="DBZH")
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


@dask.delayed
def interpolate_and_sum(arr1, arr2, conf):
    # Calculate advection correction

    oflow_method = motion.get_method("LK")
    fd_kwargs = {"buffer_mask": 10}  # avoid edge effects
    # NOTE: this line takes ~30% of the total runtime
    motion_field = oflow_method(np.log(np.stack([arr1, arr2])), fd_kwargs=fd_kwargs)

    # Calculate advection correction
    interp_frames = advection_correction.advection_correction_with_motion(
        np.array([arr1, arr2]),
        motion_field,
        T=conf["ensemble_input"]["timeres"],
        t=conf["ensemble_input"]["timeres"],
    )

    # Calculate 5 min accumulated rain rate
    arr = np.nansum(interp_frames, axis=0)
    return arr


def run(timestamp, config, ensemble_members):
    """
    Run the interpolation calculation process for given ensemble members.

    TODO:
    - take into account the snow probability when converting dBZ to rain rate

    Args:
        timestamp (str): The timestamp in the format "%Y%m%d%H%M".
        config (str): The name of the configuration file.
        ensemble_members (list): A list of ensemble members to process.

    Returns:
        None

    """
    # Read config file
    config_file = f"/config/{config}.json"
    # coef, interp_conf, snowprob_conf, input_conf, output_conf = utils.read_config(config_file)
    conf = utils.read_conf(config_file)
    curdate = datetime.strptime(timestamp, "%Y%m%d%H%M")

    # Read first file to initialize sum array and output file_dict
    timestep = conf["ensemble_input"]["timeres"]
    input_path = Path(conf["observations"]["dir"])
    first_file = input_path / conf["observations"]["filename"].format(
        timestamp=f"{timestamp}",
    )
    first_image_array, quantity, first_timestamp, gain, offset, nodata, undetect = utils.read_hdf5(
        first_file, qty="DBZH"
    )
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
        end=curdate + timedelta(minutes=conf["ensemble_input"]["fc_len"]),
        freq=f"{timestep}min",
        inclusive="right",
    ).to_pydatetime()

    accumulation_times = utils.determine_accumulation_times(curdate, conf)
    accumulation_timestep = conf["output"]["timestep"]

    data_arrays = {i: {curdate: first_arr} for i in ensemble_members}
    nodata_masks = {i: {curdate: nodata_mask_first} for i in ensemble_members}
    undetect_masks = {i: {curdate: undetect_mask_first} for i in ensemble_members}
    interp_arrays = {i: {} for i in ensemble_members}
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

        delayed_arrays = {}
        for tt in obstimes:
            logging.info(f"Reading observation {tt}")
            file = Path(conf["observations"]["dir"]) / conf["observations"]["filename"].format(
                timestamp=f"{tt:%Y%m%d%H%M}",
            )
            arr, nodata_mask, undetect_mask, file_dict_accum, lut_rr_obs, lut_sr_obs = load_file(
                file, timestep, conf, lut_rr_ens, lut_sr_ens, file_dict_accum=None
            )
            for ensno in ensemble_members:
                data_arrays[ensno][tt] = arr
                nodata_masks[ensno][tt] = nodata_mask
                undetect_masks[ensno][tt] = undetect_mask

            # Interpolate intermediate timesteps
            if conf["interp"]["interpolate"] and tt > obstimes[0]:
                arr1 = data_arrays[1][tt - timedelta(minutes=timestep)]
                arr2 = data_arrays[1][tt]

                # Calculate advection correction
                delayed_arrays[tt] = interpolate_and_sum(arr1, arr2, conf)

        # Run dask computation
        logging.info("Running dask interpolation for observations")
        arrays = dask.compute(delayed_arrays, **conf["multiprocessing"])[0]
        for tt in obstimes[1:]:
            for ensno in ensemble_members:
                interp_arrays[ensno][tt] = arrays[tt]

                nodata_mask = np.isnan(interp_arrays[ensno][tt])
                undetect_mask = interp_arrays[ensno][tt] == 0

                # Save interpolation array
                arr_ = utils.convert_dtype(
                    interp_arrays[ensno][tt],
                    conf["interpolation_output"],
                    nodata_mask,
                    undetect_mask,
                )
                # Write accumulation to file
                outfile = Path(
                    conf["interpolation_output"]["dir"]
                    + "/"
                    + conf["interpolation_output"]["filename"].format(
                        timestamp=f"{timestamp}00",
                        fc_timestep=f"{(tt - curdate).total_seconds() / 60:.0f}",
                        fc_timestamp=f"{tt:%Y%m%d%H%M}00",
                        ensno=ensno,
                        config=config,
                    )
                )
                enddate = f"{tt:%Y%m%d}"
                endtime = f"{tt:%H%M}"
                startdate = f"{tt - timedelta(minutes=timestep):%Y%m%d}"
                starttime = f"{tt - timedelta(minutes=timestep):%H%M}"

                utils.write_accumulated_h5(
                    outfile,
                    arr_,
                    file_dict_accum,
                    enddate,
                    endtime,
                    startdate,
                    starttime,
                    enddate,
                    endtime,
                    conf["interpolation_output"],
                )

            try:
                del data_arrays[1][tt - timedelta(minutes=timestep)]
            except KeyError:
                pass

    motion_fields = {}
    # Calculate advection correction for each ensemble member
    for ensno in ensemble_members:
        logging.info(f"Processing ensemble member {ensno}")
        # lue datat täällä niin ei mene niin paljoa muistia
        for i, lt in enumerate(leadtimes):
            # Read file
            file = Path(conf["ensemble_input"]["dir"]) / conf["ensemble_input"]["filename"].format(
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

        logging.info(f"Reading motion field for ensemble member {ensno}")
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

        logging.info(f"Calculating advection correction for ensemble member {ensno}")
        # Stack arrays to be interpolated so that we can calculate the interpolation only once
        adv_arrs_1 = np.stack([data_arrays[ensno][lt - timedelta(minutes=timestep)] for lt in leadtimes])
        adv_arrs_2 = np.stack([data_arrays[ensno][lt] for lt in leadtimes])

        # Do the interpolation
        # NOTE: Currently, this func takes ~50% of the total runtime
        R0 = advection_correction.interpolate_ensemble(adv_arrs_1, adv_arrs_2, motion_fields[ensno])

        # Extract interpolated frames
        for i, lt in enumerate(leadtimes):
            interp_arrays[ensno][lt] = R0[i]

            nodata_mask = np.isnan(interp_arrays[ensno][lt])
            undetect_mask = interp_arrays[ensno][lt] == 0

            # Save interpolation array
            arr_ = utils.convert_dtype(
                interp_arrays[ensno][lt],
                conf["interpolation_output"],
                nodata_mask,
                undetect_mask,
            )
            # Write accumulation to file
            outfile = Path(
                conf["interpolation_output"]["dir"]
                + "/"
                + conf["interpolation_output"]["filename"].format(
                    timestamp=f"{timestamp}00",
                    fc_timestep=f"{(lt - curdate).total_seconds() / 60:.0f}",
                    fc_timestamp=f"{lt:%Y%m%d%H%M}00",
                    ensno=ensno,
                    config=config,
                )
            )
            enddate = f"{lt:%Y%m%d}"
            endtime = f"{lt:%H%M}"
            startdate = f"{lt - timedelta(minutes=timestep):%Y%m%d}"
            starttime = f"{lt - timedelta(minutes=timestep):%H%M}"

            utils.write_accumulated_h5(
                outfile,
                arr_,
                file_dict_accum,
                enddate,
                endtime,
                startdate,
                starttime,
                enddate,
                endtime,
                conf["interpolation_output"],
            )

        # Clear memory
        del adv_arrs_1
        del adv_arrs_2
        del R0
        del data_arrays[ensno]
        del motion_fields[ensno]


if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timestamp", type=str, default="202310300845", help="Input timestamp")
    parser.add_argument(
        "--ensemble-members",
        type=int,
        nargs="+",
        default=[
            1,
        ],
        help="Ensemble members to process",
    )
    parser.add_argument("--config", type=str, default="ravake-ens", help="Config file to use.")

    options = parser.parse_args()

    # Set logging level
    logging.basicConfig(level=logging.INFO)

    run(options.timestamp, options.config, options.ensemble_members)
