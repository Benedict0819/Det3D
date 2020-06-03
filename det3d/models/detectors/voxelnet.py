from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from .. import builder
import torch


@DETECTORS.register_module
class VoxelNet(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(VoxelNet, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )

    def extract_feat(self, data):
        input_features = self.reader(data["features"], data["num_voxels"])
        x = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x = self.extract_feat(data)
        preds = self.bbox_head(x)

        if return_loss:
            return self.bbox_head.loss(example, preds)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)


@DETECTORS.register_module
class PanoviewVoxelNet(SingleStageDetector):
    def __init__(
        self,
        panoview_reader,
        reader,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        pano_feat_normalizer=None,
        pillar_feat_normalizer=None,
    ):
        super(PanoviewVoxelNet, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
        self.panoview_reader = builder.build_neck(panoview_reader)
        if pano_feat_normalizer is not None:
            self.pano_feat_normalizer = builder.build_reader(pano_feat_normalizer)
        else:
            self.pano_feat_normalizer = None

        if pillar_feat_normalizer is not None:
            self.pillar_feat_normalizer = builder.build_reader(pillar_feat_normalizer)
        else:
            self.pillar_feat_normalizer = None

    def extract_feat(self, data):
        pano_features = data["pano_features"]
        if self.pano_feat_normalizer is not None:
            pano_features = self.pano_feat_normalizer(pano_features)

        feat = data["features"]
        pano_features = self.panoview_reader(pano_features)
        n_feat = feat.shape[-1]

        pillar_features = data["features"]
        if self.pillar_feat_normalizer is not None:
            pillar_features = self.pillar_feat_normalizer(pillar_features)
            feat = torch.cat([data["features"][..., :3], pillar_features, torch.zeros(feat.shape[0], feat.shape[1], pano_features.shape[1], device=feat.device)], -1)
            n_feat += 3 # additional x, y, z (unnormalized)
        else:
            feat = torch.cat([pillar_features, torch.zeros(feat.shape[0], feat.shape[1], pano_features.shape[1], device=feat.device)], -1)

        feat[data["pt_to_voxel"][:, 0], data["pt_to_voxel"][:, 1], n_feat:] = pano_features[data['ib'], :, data['ix'], data['iy']]
        
        input_features = self.reader(feat, data["num_voxels"], coors=data["coors"], with_unnormalized_xyz = self.pano_feat_normalizer is not None)

        x = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            pano_features=example["panoview_feat"],
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
            ib=example["panoview_ib"],
            ix=example["panoview_ix"],
            iy=example["panoview_iy"],
            pt_to_voxel=example["pt_to_voxel"],
        )

        x = self.extract_feat(data)
        preds = self.bbox_head(x)

        if return_loss:
            return self.bbox_head.loss(example, preds)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)
