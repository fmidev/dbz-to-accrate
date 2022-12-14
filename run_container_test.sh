#!/bin/bash

# Build
#docker build --no-cache -t dbzh_to_acc_rate .

#CONFIG=${CONFIG:-hulehenri_case}
CONFIG=${CONFIG:-hulehenri}
#TIMESTAMP=${TIMESTAMP:-201208270000}

INPATH=/tutka/data/dev/cache/radar/fmippn/hulehenri/dbz
#INPATH=/tutka/data/input_composites/ravake
#INPATH=${INPATH:-/tutka/data/storage/radar/hulehenri}
LATEST_TIMESTAMP=`ls -t $INPATH | head -n1 | awk -F "_" '{print $1}'` #| awk -F "+" '{print $1}'`
TIMESTAMP=${TIMESTAMP:-${LATEST_TIMESTAMP}}

echo "latest timestamp:" $TIMESTAMP

#OUTPATH=${OUTPATH:-/tutka/data/storage/radar/hulehenri}
OUTPATH=${OUTPATH:-/tutka/data/dev/cache/radar/fmippn/hulehenri/accrate}
LOGPATH=${LOGPATH:-/tutka/data/dev/cache/log/fmippn/hulehenri}

echo INPATH: $INPATH
echo OUTPATH: $OUTPATH
echo LOGPATH: $LOGPATH

#Mkdirs if log and outpaths have been cleaned                                                                                                           
mkdir -p $OUTPATH
mkdir -p $LOGPATH

# Run
docker run \
       --env "timestamp=$TIMESTAMP" \
       --env "config=$CONFIG" \
       --mount type=bind,source=/tutka,target=/tutka \
       --mount type=bind,source=$INPATH,target=/input \
       --mount type=bind,source=$OUTPATH,target=/output \
       --mount type=bind,source=$LOGPATH,target=/log \
       --mount type=bind,source="$(pwd)"/config,target=/config \
       --mount type=bind,source="$(pwd)"/dbzh_to_rate.py,target=/dbzh_to_rate.py \
       --mount type=bind,source="$(pwd)"/run_dbzh_to_accr.py,target=/run_dbzh_to_accr.py \
       --mount type=bind,source="$(pwd)"/observation_dbzh_to_accr.py,target=/observation_dbzh_to_accr.py \
       --mount type=bind,source="$(pwd)"/forecast_dbzh_to_accr.py,target=/forecast_dbzh_to_accr.py \
       --mount type=bind,source="$(pwd)"/utils.py,target=/utils.py \
       dbzh_to_acc_rate:latest
