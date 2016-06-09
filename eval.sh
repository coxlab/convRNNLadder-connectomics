#!/bin/bash

export ROOT=/home/bill/git_repos/tensorflow-neuroproof-AC3/
RUNDIR=/nas/volume1/shared/bill/convRNNLadder-connectomics/runs/run_56/
#GTDIR=/nas/volume1/shared/bill/convRNNLadder-connectomics/data/
GTDIR=/home/thouis/ForBill/

echo $RUNDIR

echo creating watershed in testing set
python $ROOT/util/create_watersheds.py -i $RUNDIR/full_test_predictions_nt5.h5 -s 50 --probs_name "data" --watersheds_name "segments"

echo comparing segments
python $ROOT/util/compare_2d_segmentations.py -g $GTDIR/test_data.h5 -ngt "groundtruth_labels" -m $RUNDIR/full_test_predictions_nt5.h5 -nm "segments"
