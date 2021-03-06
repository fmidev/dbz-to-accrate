#!/bin/bash

# Build
#docker build --no-cache -t dbzh_to_acc_rate .

CONFIG=${CONFIG:-ravake}
INPATH=/tutka/data/dev/cache/radar/fmippn/nowcast2
LATEST_TIMESTAMP=`ls -t $INPATH | head -n1 | awk -F "_" '{print $1}' | awk -F "+" '{print $1}'`
TIMESTAMP=${TIMESTAMP:-${LATEST_TIMESTAMP}}

echo "latest timestamp:" $TIMESTAMP

OUTPATH=/tutka/data/dev/cache/radar/fmippn/postprocess
LOGPATH=/tutka/data/dev/cache/log/fmippn/postprocess

echo INPATH: $INPATH
echo OUTPATH: $OUTPATH
echo LOGPATH: $LOGPATH

#Mkdirs if log and outpaths have been cleaned                                                                                                           
mkdir -p $OUTPATH
mkdir -p $LOGPATH

#--mount type=bind,source=/mnt/meru/data/prod/radman/mallidata,target=/nwp_input \

# Run
docker run \
        --env "timestamp=$TIMESTAMP" \
        --env "config=$CONFIG" \
       	--mount type=bind,source=$INPATH,target=/input \
	--mount type=bind,source=$OUTPATH,target=/output \
	--mount type=bind,source=$LOGPATH,target=/log \
	--mount type=bind,source="$(pwd)"/dbzh_to_acc_rate.py,target=/dbzh_to_acc_rate.py \
	--mount type=bind,source="$(pwd)"/dbzh_to_rate.py,target=/dbzh_to_rate.py \
	--mount type=bind,source="$(pwd)"/config_dbzhtorate_ravake.json,target=/config_dbzhtorate_ravake.json \
	dbzh_to_acc_rate:latest
