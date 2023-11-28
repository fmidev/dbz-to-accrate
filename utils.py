import h5py
import hiisi
import numpy as np
import gzip
import math
import json
import matplotlib.pyplot as plt
import logging
import sys
import pandas as pd
from datetime import timedelta


def read_config(config_file):
    """Read parameters from config file.

    Keyword arguments:
    config_file -- json file containing input parameters

    Return:
    coef -- dictionary containing coefficients
    input_conf -- dictionary containing input parameters
    output_conf -- dictionary containing output parameters

    """

    with open(config_file, "r") as jsonfile:
        data = json.load(jsonfile)

    coef = data["coef"]
    interp_conf = data["interp"]
    snowprob_conf = data["snowprob"]
    input_conf = data["input"]
    output_conf = data["output"]

    return coef, interp_conf, snowprob_conf, input_conf, output_conf


def read_conf(config_file):
    """Read parameters from config file.

    Keyword arguments:
    config_file -- json file containing input parameters

    Return:
    dict -- dictionary containing configuration

    """
    with open(config_file, "r") as jsonfile:
        data = json.load(jsonfile)

    return data


def read_hdf5(image_h5_file, qty="DBZH"):
    """Read image array from ODIM hdf5 file.

    Keyword arguments:
    image_h5_file -- ODIM hdf5 file
    qty -- array quantity that is read

    Return:
    image_array -- numpy array containing DBZH or RATE array
    quantity -- array quantity
    timestamp -- timestamp of image_array
    mask_nodata -- masked array where image_array has nodata value
    gain -- gain of image_array
    offset -- offset of image_array

    """

    # Read RATE or DBZH from hdf5 file
    logging.info(f"Extracting data from {image_h5_file} file")
    comp = hiisi.OdimCOMP(image_h5_file, "r")

    test = comp.select_dataset(qty)
    if test is not None:
        image_array = comp.dataset
        quantity = qty
    else:
        logging.error(f"{qty} array not found in the file {image_h5_file}!")
        raise ValueError(f"{qty} array not found in the file {image_h5_file}!")

    # Read nodata and undetect values from metadata for masking
    gen = comp.attr_gen("nodata")
    pair = gen.__next__()
    nodata = pair.value
    gen = comp.attr_gen("undetect")
    pair = gen.__next__()
    undetect = pair.value

    # Read gain and offset values from metadata
    gen = comp.attr_gen("gain")
    pair = gen.__next__()
    gain = pair.value
    gen = comp.attr_gen("offset")
    pair = gen.__next__()
    offset = pair.value

    # Read timestamp from metadata
    gen = comp.attr_gen("date")
    pair = gen.__next__()
    date = pair.value
    gen = comp.attr_gen("time")
    pair = gen.__next__()
    time = pair.value

    timestamp = date + time

    return image_array, quantity, timestamp, gain, offset, int(nodata), int(undetect)


def read_motion_hdf5(motion_h5_file):
    """Read motion field array from ODIM hdf5 file.

    Keyword arguments:
    motion_h5_file -- ODIM hdf5 file containing DBZH or RATE array

    Return:
    amv -- numpy array containing motion field of shape (2, height, width)
    timestep -- timestep of motion field in minutes
    kmperpixel -- kmperpixel of motion field

    """

    # Read RATE or DBZH from hdf5 file
    logging.info(f"Extracting data from {motion_h5_file} file")
    comp = hiisi.OdimCOMP(motion_h5_file, "r")

    # Read AMVU and AMVV from hdf5 file
    test = comp.select_dataset("AMVU")
    if test is not None:
        amvu = comp.dataset
    else:
        logging.error(f"AMVU array not found in the file {motion_h5_file}!")
        sys.exit(1)

    test = comp.select_dataset("AMVV")
    if test is not None:
        amvv = comp.dataset
    else:
        logging.error(f"AMVV array not found in the file {motion_h5_file}!")
        sys.exit(1)

    # Get motion field timestep
    gen = comp.attr_gen("input_interval")
    pair = gen.__next__()
    timestep = pair.value / 60

    # Get motion field kmperpixel
    gen = comp.attr_gen("kmperpixel")
    pair = gen.__next__()
    kmperpixel = pair.value

    return np.stack([amvu, amvv]), timestep, kmperpixel


def convert_dtype(accumulated_image, output_conf, nodata_mask, undetect_mask):
    """Change output data dtype (e.g. to 16 bit unsigned integer) and rescale data if needed

    Keyword arguments:
    accumulated_image --
    output_conf --
    nodata_mask --
    undetect_mask --

    Return:
    scaled_image_new_dtype --

    """
    scaled_image = (accumulated_image - output_conf["offset"]) / output_conf["gain"]
    scaled_image[nodata_mask] = output_conf["nodata"]
    scaled_image[undetect_mask] = output_conf["undetect"]
    scaled_image_new_dtype = scaled_image.astype(output_conf["dtype"])

    return scaled_image_new_dtype


def unpack_dtype(scaled_image, output_conf):
    """Unpack scaled data to original dtype

    Keyword arguments:
    scaled_image --
    output_conf --

    Return:
    unpacked_image --

    """
    nodata_mask = scaled_image == output_conf["nodata"]
    undetect_mask = scaled_image == output_conf["undetect"]
    unpacked_image = scaled_image * output_conf["gain"] + output_conf["offset"]

    unpacked_image[nodata_mask] = np.nan
    unpacked_image[undetect_mask] = 0

    return unpacked_image, nodata_mask, undetect_mask


def init_filedict_accumulation(image_h5_file):
    """Copy common metadata from input file to be used as
    output file template.

    Keyword arguments:
    image_h5_file -- ODIM hdf5 file containing DBZH or RATE array

    Return:
    file_dict_accum -- dictionary containing ODIM metadata

    """

    # Read metadata from image h5 file
    comp = hiisi.OdimCOMP(image_h5_file, "r")

    # Write metadata to file_dict
    file_dict_accum = {
        "/": dict(comp["/"].attrs.items()),
        "/how": dict(comp["/how"].attrs.items()),
        "/where": dict(comp["/where"].attrs.items()),
    }

    return file_dict_accum


def write_accumulated_h5(
    output_h5,
    accumulated_image,
    file_dict_accum,
    date,
    time,
    startdate,
    starttime,
    enddate,
    endtime,
    output_conf,
    quantity="ACRR",
):
    """Write accumulated precipitation rate to ODIM hdf5 file.

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
    output_conf

    """

    # Insert date and time to file_dict
    file_dict_accum["/what"] = {
        "date": date,
        "object": np.string_("COMP"),
        "source": np.string_("ORG:247"),
        "time": time,
        "version": np.string_("H5rad 2.0"),
    }
    # Insert startdate and -time and enddate- and time
    file_dict_accum["/dataset1/data1/what"] = {
        "gain": output_conf["gain"],
        "nodata": output_conf["nodata"],
        "offset": output_conf["offset"],
        "product": np.string_("COMP"),
        "quantity": np.string_(quantity),
        "undetect": output_conf["undetect"],
        "startdate": startdate,
        "starttime": starttime,
        "enddate": enddate,
        "endtime": endtime,
    }
    # Insert accumulated dataset into file_dict
    file_dict_accum["/dataset1/data1/data"] = {
        "DATASET": accumulated_image,
        "COMPRESSION": "gzip",
        "COMPRESSION_OPTS": 6,
        "CLASS": np.string_("IMAGE"),
        "IMAGE_VERSION": np.string_("1.2"),
    }
    # Write hdf5 file from file_dict
    with hiisi.HiisiHDF(output_h5, "w") as h:
        h.create_from_filedict(file_dict_accum)


def plot_array(array, outfile):
    plt.figure()
    plt.imshow(array)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    plt.savefig(outfile)
    plt.close()


def _convert_motion_units_pix2ms(data_pxts, kmperpixel=1.0, timestep=1.0):
    """Convert atmospheric motion vectors from pixel/timestep units to m/s.

    Input:
        data_pxts -- motion vectors in "pixels per timestep" units
        kmperpixel -- kilometers in pixel
        timestep -- timestep lenght in minutes

    Output:
        data_ms -- motion vectors in m/s units
    """
    meters_per_pixel = kmperpixel * 1000
    seconds_in_timestep = timestep * 60
    # data unit conversion logic:
    # px_per_timestep = px_per_s * s_per_timestep
    #                 = px_per_km * km_per_s * s_per_timestep
    #                 = px_per_km * km_per_m * m_per_s * s_per_timestep
    # Solve for m_per_s:
    # m_per_s = px_per_timestep / (px_per_km * km_per_m * s_per_timestep)
    #         = px_per_timestep * km_per_px * m_per_km / s_per_timestep
    #         = px_per_timestep * meters_per_pixel / seconds_in_timestep
    data_ms = data_pxts * meters_per_pixel / seconds_in_timestep
    return data_ms


def _convert_motion_units_ms2pix(data_ms, kmperpixel=1.0, timestep=1.0):
    meters_per_pixel = kmperpixel * 1000
    seconds_in_timestep = timestep * 60
    data_pxts = data_ms * (1 / meters_per_pixel) * seconds_in_timestep
    return data_pxts


def determine_accumulation_times(curdate, conf):
    """
    Determine the accumulation times for a given date.

    Args:
        curdate (datetime): The current date.
        conf (dict): The configuration settings.

    Returns:
        numpy.ndarray: An array of accumulation times.

    """
    accumulation_timestep = conf["output"]["timestep"]

    if conf["output"]["write_acrr_fixed_step"]:
        accumulation_times_fixed = pd.date_range(
            start=pd.Timestamp(curdate).floor(f"{accumulation_timestep}min"),
            end=curdate + timedelta(minutes=conf["ensemble_input"]["fc_len"]),
            freq=f"{accumulation_timestep}min",
            inclusive="right",
        ).to_pydatetime()
    else:
        accumulation_times_fixed = np.array([])
    if conf["output"]["write_acrr_from_start"]:
        accumulation_times_from_start = pd.date_range(
            start=curdate,
            end=curdate + timedelta(minutes=conf["ensemble_input"]["fc_len"]),
            freq=f"{accumulation_timestep}min",
            inclusive="right",
        ).to_pydatetime()
    else:
        accumulation_times_from_start = np.array([])

    accumulation_times = np.unique(np.concatenate([accumulation_times_fixed, accumulation_times_from_start]))
    accumulation_times.sort()
    return accumulation_times
