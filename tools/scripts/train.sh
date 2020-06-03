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

# CGBS 10sweep
# python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py examples/cbgs/configs/nusc_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn_10sweep.py --work_dir=$NUSC_CBGS_WORK_DIR --load_from /home/xiac/train_out/Det3D_Outputs/NUSC_CBGS_10sweep_20200226-043301/latest.pth

# Pano and PFPN 10 sweep
# python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py examples/point_pillars/configs/nusc_all_point_pillars_image_view_pfpn_mghead_syncbn_10sweep.py --work_dir=$PP_WORK_DIR/ --resume_from /home/xiac/train_out/Det3D_Outputs/PointPillars_10_sweep_all_points_20200222-124441/latest.pth
# --resume_from /home/xiac/train_out/Det3D_Outputs/PointPillars_10_sweep_1_frame_20200221-055708/latest.pth
# --load_from /home/xiac/train_out/Det3D_Outputs/PointPillars_pano_pfpn_min4_range_sorted_20200211-070541/latest.pth

# CGBS pano
# python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py examples/cbgs/configs/nusc_all_vfev3_pano_spmiddleresnetfhd_rpn2_mghead_syncbn_1sweep.py --work_dir=$NUSC_CBGS_WORK_DIR --load_from /home/xiac/train_out/Det3D_Outputs/NUSC_CBGS_pano_1sweep_64ch_0_0002_20200224-212538/latest.pth

# Pano and PFPN 10 sweep 10x10
# python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py examples/point_pillars/configs/nusc_all_point_pillars_image_view_pfpn_mghead_syncbn_10sweep_10x10.py --work_dir=$PP_WORK_DIR/ --load_from /home/xiac/train_out/Det3D_Outputs/PointPillars_pano_10sweep_0_1m_20200229-191824/latest.pth

# CGBS pano 10sweep
# python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py examples/cbgs/configs/nusc_all_vfev3_pano_spmiddleresnetfhd_rpn2_mghead_syncbn_10sweep.py --work_dir=$NUSC_CBGS_WORK_DIR --load_from /home/xiac/train_out/Det3D_Outputs/NUSC_CBGS_10_sweep_retraining_20200229-002057/latest.pth

# CGBS pano 10sweep improved
# python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py examples/cbgs/configs/nusc_all_vfev3_pano_spmiddleresnetfhd_rpn2_mghead_syncbn_imp_10sweep.py --work_dir=$NUSC_CBGS_WORK_DIR --load_from /home/xiac/train_out/Det3D_Outputs/NUSC_CBGS_pano_10sweep_20200301-083612/latest.pth

# CGBS pano 10sweep highres
# python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py examples/cbgs/configs/nusc_all_vfev3_pano_spmiddleresnetfhd_rpn2_mghead_syncbn_highres_10sweep.py --work_dir=$NUSC_CBGS_WORK_DIR --resume_from /home/xiac/train_out/Det3D_Outputs/NUSC_CBGS_pano_10sweep_highres_20200305-011042/latest.pth

# CGBS pano 10sweep highres improved add
# python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py examples/cbgs/configs/nusc_all_vfev3_pano_spmiddleresnetfhd_rpn2_mghead_syncbn_highres_imp_10sweep.py --work_dir=$NUSC_CBGS_WORK_DIR --resume_from /home/xiac/train_out/Det3D_Outputs/NUSC_CBGS_pano_10sweep_imp_add_20200315-213308/latest.pth

# CGBS pano 10sweep highres improved
# python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py examples/cbgs/configs/nusc_all_vfev3_pano_spmiddleresnetfhd_rpn2_mghead_syncbn_highres_imp_10sweep.py --work_dir=$NUSC_CBGS_WORK_DIR --resume_from /home/xiac/train_out/Det3D_Outputs/NUSC_CBGS_pano_imp_highres_with_speed_20200330-233735/latest.pth

# CGBS pano 10sweep highres improved and high vertical improved
python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py examples/cbgs/configs/nusc_all_vfev3_pano_spmiddleresnetfhd_rpn2_mghead_syncbn_highres_highvert_imp_10sweep.py --work_dir=$NUSC_CBGS_WORK_DIR --resume_from /home/xiac/train_out/Det3D_Outputs/NUSC_CBGS_high_vert_3553_20200406-193614/latest.pth