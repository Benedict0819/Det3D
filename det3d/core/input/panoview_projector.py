import numpy as np
from det3d.ops.point_cloud.point_cloud_ops import points_to_panoview


class PanoviewProjector:
    def __init__(self, 
                 lidar_xyz=[0, 0, 0], 
                 h_steps=(-180, 180, 0.2), 
                 v_steps=(-30, 10, 1),
                 sort_points_by_range=False,
                 prioritize_key_frame=False):
        self.h_steps = h_steps
        self.v_steps = v_steps
        self.lidar_xyz = lidar_xyz
        self.sort_points_by_range = sort_points_by_range
        self.prioritize_key_frame = prioritize_key_frame
        self.h_len = np.rint((self.h_steps[1] - self.h_steps[0]) / self.h_steps[2]).astype(np.int)
        self.v_len = np.rint((self.v_steps[1] - self.v_steps[0]) / self.v_steps[2]).astype(np.int)
        self.proj_shape = np.array([self.h_len, self.v_len])


    def project(self, points):
        return points_to_panoview(
            points,
            self.lidar_xyz,
            self.h_steps,
            self.v_steps,
            self.sort_points_by_range,
            self.prioritize_key_frame,
        )

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self):
        return self._max_num_points

    @property
    def point_cloud_range(self):
        return self._point_cloud_range

    @property
    def grid_size(self):
        return self._grid_size
