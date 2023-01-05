from mmdeploy.core import FUNCTION_REWRITER
from mmcv.ops.sparse_structure import SparseConvTensor
from mmcv.ops import sparse_ops as ops
from mmcv.ops import sparse_functional as Fsp
import torch
from typing import Any


class ONNXSparseConvop(torch.autograd.Function):
    """Sparse Convolution.

    Please refer to `SECOND <https://www.mdpi.com/1424-8220/18/10/3337>`_ for
    more details.
    """

    @staticmethod
    def forward(ctx: Any, features: torch.Tensor, filters: torch.nn.Parameter,
                indice_pairs: torch.Tensor, indice_pair_num: torch.Tensor,
                num_activate_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): Features that needs to convolute.
            filters (torch.nn.parameter.Parameter): Convolution filters.
            indice_pairs (torch.Tensor): Indice pairs between inputs locations
                and outputs locations.
            indice_pair_num (torch.Tensor): Indice pairs num.
            num_activate_out (torch.Tensor): Output channels num.

        Returns:
            torch.Tensor: Output features from gather-gemm-scatter.
        """
        ctx.save_for_backward(indice_pairs, indice_pair_num, features, filters)
        return ops.indice_conv(features, filters, indice_pairs,
                               indice_pair_num, num_activate_out, False)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        indice_pairs, indice_pair_num, features, filters = ctx.saved_tensors
        input_bp, filters_bp = ops.indice_conv_backward(
            features, filters, grad_output, indice_pairs, indice_pair_num,
            False)

        return input_bp, filters_bp, None, None, None

    @staticmethod
    def symbolic(g, features, filters, indice_pairs, indice_pair_num, num_activate_out):
        return g.op(
            'mmdeploy::SparseConvFunction',
            features,
            filters,
            indice_pairs,
            indice_pair_num,
            num_activate_out
        )


class ONNXSubMConvop(torch.autograd.Function):
    """Create onnx::SubMConvFunction op."""

    @staticmethod
    def forward(ctx: Any, features: torch.Tensor, filters: torch.nn.Parameter, 
                indice_pairs: torch.Tensor, indice_pair_num: torch.Tensor,
                num_activate_out: torch.Tensor) -> torch.Tensor:

        """
        Args:
            features (torch.Tensor): Features that needs to convolute.
            filters (torch.nn.parameter.Parameter): Convolution filters.
            indice_pairs (torch.Tensor): Indice pairs between inputs locations
                and outputs locations.
            indice_pair_num (torch.Tensor): Indice pairs num.
            num_activate_out (torch.Tensor): Output channels num.

        Returns:
            torch.Tensor: Output features from gather-gemm-scatter.
        """
        ctx.save_for_backward(indice_pairs, indice_pair_num, features, filters)
        return ops.indice_conv(features, filters, indice_pairs,
                               indice_pair_num, num_activate_out, False, True)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        indice_pairs, indice_pair_num, features, filters = ctx.saved_tensors
        input_bp, filters_bp = ops.indice_conv_backward(
            features, filters, grad_output, indice_pairs, indice_pair_num,
            False, True)

        return input_bp, filters_bp, None, None, None
    
    @staticmethod
    def symbolic(g, features, filters, indice_pairs, indice_pair_num, num_activate_out):
        return g.op(
            'mmdeploy::SubMConvFunction',
            features,
            filters,
            indice_pairs,
            indice_pair_num,
            num_activate_out
        )

indice_conv = ONNXSparseConvop.apply
indice_subm_conv = ONNXSubMConvop.apply

@FUNCTION_REWRITER.register_rewriter(
    'mmcv.ops.sparse_conv.SparseConvolution.forward',
     backend='tensorrt')
def sparse_convolution__forward(self, input):
    assert isinstance(input, SparseConvTensor)
    features = input.features
    device = features.device
    indices = input.indices
    spatial_shape = input.spatial_shape
    batch_size = input.batch_size
    if not self.subm:
        if self.transposed:
            out_spatial_shape = ops.get_deconv_output_size(
                spatial_shape, self.kernel_size, self.stride, self.padding,
                self.dilation, self.output_padding)
        else:
            out_spatial_shape = ops.get_conv_output_size(
                spatial_shape, self.kernel_size, self.stride, self.padding,
                self.dilation)

    else:
        out_spatial_shape = spatial_shape

    if self.conv1x1:
        features = torch.mm(
            input.features,
            self.weight.view(self.in_channels, self.out_channels))
        if self.bias is not None:
            features += self.bias
        out_tensor = SparseConvTensor(features, input.indices,
                                        input.spatial_shape,
                                        input.batch_size)
        out_tensor.indice_dict = input.indice_dict
        out_tensor.grid = input.grid
        return out_tensor
    data = input.find_indice_pair(self.indice_key)
    if self.inverse:
        assert data is not None and self.indice_key is not None
        _, outids, indice_pairs, indice_pair_num, out_spatial_shape = data
        assert indice_pairs.shape[0] == np.prod(
            self.kernel_size
        ), 'inverse conv must have same kernel size as its couple conv'
    else:
        if self.indice_key is not None and data is not None:
            outids, _, indice_pairs, indice_pair_num, _ = data
        else:
            outids, indice_pairs, indice_pair_num = ops.get_indice_pairs(
                indices,
                batch_size,
                spatial_shape,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                self.output_padding,
                self.subm,
                self.transposed,
                grid=input.grid)
            input.indice_dict[self.indice_key] = (outids, indices,
                                                    indice_pairs,
                                                    indice_pair_num,
                                                    spatial_shape)
    if self.fused_bn:
        assert self.bias is not None
        out_features = ops.fused_indice_conv(features, self.weight,
                                                self.bias,
                                                indice_pairs.to(device),
                                                indice_pair_num,
                                                outids.shape[0], self.inverse,
                                                self.subm)
    else:
        if self.subm:
            out_features = indice_subm_conv(features, self.weight,
                                                indice_pairs.to(device),
                                                indice_pair_num,
                                                outids.shape[0])
        else:
            if self.inverse:
                out_features = Fsp.indice_inverse_conv(
                    features, self.weight, indice_pairs.to(device),
                    indice_pair_num, outids.shape[0])
            else:
                out_features = indice_conv(features, self.weight,
                                                indice_pairs.to(device),
                                                indice_pair_num,
                                                outids.shape[0])

        if self.bias is not None:
            out_features += self.bias
    out_tensor = SparseConvTensor(out_features, outids, out_spatial_shape,
                                    batch_size)
    out_tensor.indice_dict = input.indice_dict
    out_tensor.grid = input.grid
    return out_tensor
