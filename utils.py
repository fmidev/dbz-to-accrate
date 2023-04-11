import h5py
import hiisi
import numpy as np
import gzip
import math
import json
import matplotlib.pyplot as plt
import logging
import sys


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


def read_hdf5(image_h5_file):
    """Read image array from ODIM hdf5 file.

    Keyword arguments:
    image_h5_file -- ODIM hdf5 file containing DBZH or RATE array

    Return:
    image_array -- numpy array containing DBZH or RATE array
    quantity -- array quantity, either 'DBZH' or 'RATE'
    timestamp -- timestamp of image_array
    mask_nodata -- masked array where image_array has nodata value
    gain -- gain of image_array
    offset -- offset of image_array

    """

    # Read RATE or DBZH from hdf5 file
    logging.info(f"Extracting data from {image_h5_file} file")
    comp = hiisi.OdimCOMP(image_h5_file, "r")
    # Read RATE array if found in dataset
    test = comp.select_dataset("DBZH")
    if test is not None:
        image_array = comp.dataset
        quantity = "DBZH"
    else:
        # Look for RATE array
        test = comp.select_dataset("RATE")
        if test is not None:
            image_array = comp.dataset
            quantity = "RATE"
        else:
            # Look for SNOWPROB array
            test = comp.select_dataset("SNOWPROB")
            if test is not None:
                image_array = comp.dataset
                quantity = "SNOWPROB"
            else:
                logging.error(
                    f"DBZH, RATE or SNOWPROB array not found in the file {image_h5_file}!"
                )
                sys.exit(1)

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
        "quantity": np.string_("ACRR"),
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
