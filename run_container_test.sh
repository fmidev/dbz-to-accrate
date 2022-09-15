#!/bin/bash

# Build
#docker build --no-cache -t dbzh_to_acc_rate .

#CONFIG=${CONFIG:-ravake_composite}
CONFIG=${CONFIG:-hulehenri_composite}
FILETYPE=${FILETYPE:-composite}
TIMESTAMP=${TIMESTAMP:-201208270000}

#INPATH=/tutka/data/dev/cache/radar/fmippn/ravake
#INPATH=/tutka/data/input_composites/ravake
INPATH=${INPATH:-/tutka/data/storage/radar/hulehenri}
#LATEST_TIMESTAMP=`ls -t $INPATH | head -n1 | awk -F "_" '{print $1}' | awk -F "+" '{print $1}'`
#TIMESTAMP=${TIMESTAMP:-${LATEST_TIMESTAMP}}

echo "latest timestamp:" $TIMESTAMP

OUTPATH=${OUTPATH:-/tutka/data/storage/radar/hulehenri}
LOGPATH=${LOGPATH:-/tutka/data/dev/cache/log/fmippn/hulehenri}

echo INPATH: $INPATH
echo OUTPATH: $OUTPATH
echo LOGPATH: $LOGPATH

#Mkdirs if log and outpaths have been cleaned                                                                                                           
mkdir -p $OUTPATH
mkdir -p $LOGPATH

#--mount type=bind,source=/mnt/meru/data/prod/radman/mallidata,target=/nwp_input \
#--user 7939:5008 \

# Run
docker run \
       --env "timestamp=$TIMESTAMP" \
       --env "config=$CONFIG" \
       --mount type=bind,source=$INPATH,target=/input \
       --mount type=bind,source=$OUTPATH,target=/output \
       --mount type=bind,source=$LOGPATH,target=/log \
       --mount type=bind,source="$(pwd)"/fmippn_dbzh_to_accr.py,target=/fmippn_dbzh_to_accr.py \
       --mount type=bind,source="$(pwd)"/composite_dbzh_to_accr.py,target=/composite_dbzh_to_accr.py \
       --mount type=bind,source="$(pwd)"/utils.py,target=/utils.py \
       --mount type=bind,source="$(pwd)"/config/config_dbzhtorate_ravake.json,target=/config/config_dbzhtorate_ravake.json \
       --mount type=bind,source="$(pwd)"/config/config_dbzhtorate_ravake_composite.json,target=/config/config_dbzhtorate_ravake_composite.json \
       --mount type=bind,source="$(pwd)"/config/config_dbzhtorate_hulehenri_composite.json,target=/config/config_dbzhtorate_hulehenri_composite.json \
       dbzh_to_acc_rate:latest
