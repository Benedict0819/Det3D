TASK_DESC=$1
DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`
OUT_DIR=/home/xiac/train_out/Det3D_Outputs

NUSC_CBGS_WORK_DIR=$OUT_DIR/NUSC_CBGS_$TASK_DESC\_$DATE_WITH_TIME
LYFT_CBGS_WORK_DIR=$OUT_DIR/LYFT_CBGS_$TASK_DESC\_$DATE_WITH_TIME
SECOND_WORK_DIR=$OUT_DIR/SECOND_$TASK_DESC\_$DATE_WITH_TIME
PP_WORK_DIR=$OUT_DIR/PointPillars_$TASK_DESC\_$DATE_WITH_TIME

if [ ! $TASK_DESC ] 
then
    echo "TASK_DESC must be specified."
    echo "Usage: train.sh task_description"
    exit $E_ASSERT_FAILED
fi

# Voxelnet
# python -m torch.distributed.launch --nproc_per_node=8 ./tools/train.py examples/second/configs/kitti_car_vfev3_spmiddlefhd_rpn1_mghead_syncbn.py --work_dir=$SECOND_WORK_DIR/
# python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py examples/cbgs/configs/nusc_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn.py --work_dir=$NUSC_CBGS_WORK_DIR
# python -m torch.distributed.launch --nproc_per_node=8 ./tools/train.py examples/second/configs/lyft_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn.py --work_dir=$LYFT_CBGS_WORK_DIR

# python -W once -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py examples/point_pillars/configs/nusc_all_point_pillars_mghead_syncbn_1sweep.py --work_dir=$PP_WORK_DIR/

# --resume_from=/home/xiac/train_out/Det3D_Outputs/PointPillars_4gpu1sweep_syncbn_sgd_20200131-195801/latest.pth
# --load_from=/home/xiac/train_out/Det3D_Outputs/PointPillars_bn_default_20/epoch_20.pt

# 10 sweep
# python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py examples/point_pillars/configs/nusc_all_point_pillars_mghead_syncbn.py --work_dir=$PP_WORK_DIR/


# PointPillars
# python -m torch.distributed.launch --nproc_per_node=8 ./tools/train.py ./examples/point_pillars/configs/original_pp_mghead_syncbn_kitti.py --work_dir=$PP_WORK_DIR

# python -X faulthandler ./tools/train.py examples/point_pillars/configs/nusc_all_point_pillars_mghead_syncbn_1sweep.py --work_dir=$PP_WORK_DIR/ 


# PointPillars with PFPN
# python -W once -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py examples/point_pillars/configs/nusc_all_point_pillars_pfpn_mghead_syncbn_1sweep.py --work_dir=$PP_WORK_DIR/ 


# Pano
# python -W once -m torch.distributed.launch --nproc_per_node=1 ./tools/train.py examples/point_pillars/configs/nusc_all_point_pillars_image_view_mghead_syncbn_1sweep.py --work_dir=$PP_WORK_DIR/ 

# Pano and PFPN
# python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py examples/point_pillars/configs/nusc_all_point_pillars_image_view_pfpn_mghead_syncbn_1sweep.py --work_dir=$PP_WORK_DIR/ --load_from /home/xiac/train_out/Det3D_Outputs/PointPillars_25x25_corrected_with_ring_20200217-220746/latest.pth
# /home/xiac/train_out/Det3D_Outputs/PointPillars_pano_pfpn_min4_range_sorted_20200211-070541/det3d/

# CGBS 1sweep
# python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py examples/cbgs/configs/nusc_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn_1sweep.py --work_dir=$NUSC_CBGS_WORK_DIR --resume_from /home/xiac/train_out/Det3D_Outputs/NUSC_CBGS_1sweep_20200211-135711/latest.pth

# Pano and PFPN 10 sweep
python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py examples/point_pillars/configs/nusc_all_point_pillars_image_view_pfpn_mghead_syncbn_10sweep.py --work_dir=$PP_WORK_DIR/ --resume_from /home/xiac/train_out/Det3D_Outputs/PointPillars_10_sweep_1_frame_20200221-055708/latest.pth
# --load_from /home/xiac/train_out/Det3D_Outputs/PointPillars_pano_pfpn_min4_range_sorted_20200211-070541/latest.pth
