import torch
from mmcv.utils import ext_loader
from mmdeploy.core import FUNCTION_REWRITER

ext_module = ext_loader.load_ext('_ext', [
    'get_indice_pairs_2d_forward', 'get_indice_pairs_3d_forward',
    'get_indice_pairs_4d_forward', 'get_indice_pairs_2d_backward',
    'get_indice_pairs_3d_backward', 'indice_conv_forward',
    'indice_conv_backward', 'fused_indice_conv_forward',
    'indice_maxpool_forward', 'indice_maxpool_backward'
])

@FUNCTION_REWRITER.register_rewriter(
    func_name='mmcv.ops.sparse_ops.get_indice_pairs',
    backend='tensorrt')
def get_indice__pairs(
                     indices,
                     batch_size,
                     spatial_shape,
                     ksize=3,
                     stride=1,
                     padding=0,
                     dilation=1,
                     out_padding=0,
                     subm=False,
                     transpose=False,
                     grid=None):
    from mmcv.ops.sparse_ops import get_conv_output_size
    ndim = indices.shape[1] - 1
    if not isinstance(ksize, (list, tuple)):
        ksize = [ksize] * ndim
    if not isinstance(stride, (list, tuple)):
        stride = [stride] * ndim
    if not isinstance(padding, (list, tuple)):
        padding = [padding] * ndim
    if not isinstance(dilation, (list, tuple)):
        dilation = [dilation] * ndim
    if not isinstance(out_padding, (list, tuple)):
        out_padding = [out_padding] * ndim

    for d, s in zip(dilation, stride):
        assert any([s == 1, d == 1]), "don't support this."

    if not subm:
        if transpose:
            out_shape = get_deconv_output_size(spatial_shape, ksize, stride,
                                               padding, dilation, out_padding)
        else:
            out_shape = get_conv_output_size(spatial_shape, ksize, stride,
                                             padding, dilation)

    else:
        out_shape = spatial_shape
    if grid is None:
        get_indice_pairs_func = ext_module.get_indice_pairs_3d_forward
        return get_indice_pairs_func(indices, batch_size, out_shape,
                                     spatial_shape, ksize, stride, padding,
                                     dilation, out_padding, int(subm),
                                     int(transpose))
    else:
        get_indice_pairs_func = ext_module.get_indice_pairs_3d_backward
        return get_indice_pairs_func(indices, grid, batch_size, out_shape,
                                     spatial_shape, ksize, stride, padding,
                                     dilation, out_padding, int(subm),
                                     int(transpose))



