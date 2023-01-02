from typing import Tuple
  
from torch import Tensor
from mmdeploy.core import FUNCTION_REWRITER

@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.voxelnet.VoxelNet.extract_feat',
    backend='tensorrt')
def voxelnet_extract__feat(self, batch_inputs_dict: dict) -> Tuple[Tensor]:
    """Extract features from points."""
    voxel_dict = batch_inputs_dict['voxels']
    voxel_features = self.voxel_encoder(voxel_dict['voxels'],
                                        voxel_dict['num_points'],
                                        voxel_dict['coors'])
    batch_size = voxel_dict['coors'][-1, 0] + 1
    x = self.middle_encoder(voxel_features, voxel_dict['coors'],
                            batch_size)
    x = self.backbone(x)
    if self.with_neck:
        x = self.neck(x)
    return x                