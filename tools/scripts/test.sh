#!/bin/bash
# CONFIG=$1
# WORK_DIR=$2
# CHECKPOINT=$3

OUT_DIR=/home/xiac/train_out/Det3D_Outputs

NUSC_CBGS_WORK_DIR=$OUT_DIR/NUSC_CBGS_$TASK_DESC\_$DATE_WITH_TIME
LYFT_CBGS_WORK_DIR=$OUT_DIR/LYFT_CBGS_$TASK_DESC\_$DATE_WITH_TIME
SECOND_WORK_DIR=$OUT_DIR/SECOND_$TASK_DESC\_$DATE_WITH_TIME
PP_WORK_DIR=$OUT_DIR/PointPillars_$TASK_DESC\_$DATE_WITH_TIME

WORK_DIR=$OUT_DIR/eval

# CONFIG=examples/point_pillars/configs/nusc_all_point_pillars_image_view_mghead_syncbn_1sweep.py
# CHECKPOINT=/home/xiac/train_out/Det3D_Outputs/PointPillars_pano_20200205-004526/latest.pth

# CHECKPOINT_PATH=/home/xiac/train_out/Det3D_Outputs/PointPillars_pano_pfpn_min4_range_sorted_20200211-070541
# CHECKPOINT_PATH=/home/xiac/train_out/Det3D_Outputs/NUSC_CBGS_pano_imp_highres_with_speed_corrected_angle_20200401-074135
# CHECKPOINT=$CHECKPOINT_PATH/epoch_4.pth



CHECKPOINT_PATH=/home/xiac/train_out/Det3D_Outputs/NUSC_CBGS_high_vert_20200410-061813
CHECKPOINT=$CHECKPOINT_PATH/epoch_10.pth

# CONFIG=$CHECKPOINT_PATH/det3d/examples/point_pillars/configs/nusc_all_point_pillars_image_view_pfpn_mghead_syncbn_10sweep.py
CONFIG=examples/cbgs/configs/nusc_all_vfev3_pano_spmiddleresnetfhd_rpn2_mghead_syncbn_highres_highvert_imp_10sweep.py
# CONFIG=examples/cbgs/configs/nusc_all_vfev3_pano_spmiddleresnetfhd_rpn2_mghead_syncbn_highres_highvert_imp_10sweep.py
# CONFIG=$CHECKPOINT_PATH/det3d/examples/point_pillars/configs/nusc_all_vfev3_pano_spmiddleresnetfhd_rpn2_mghead_syncbn_highres_highvert_imp_10sweep.py


# Test
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    ./tools/dist_test.py \
    $CONFIG \
    --work_dir=$WORK_DIR \
    --checkpoint=$CHECKPOINT \


