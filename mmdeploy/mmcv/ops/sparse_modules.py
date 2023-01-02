from mmdeploy.core import FUNCTION_REWRITER
import torch
from mmcv.ops.sparse_structure import SparseConvTensor

@FUNCTION_REWRITER.register_rewriter(
    func_name='mmcv.ops.sparse_modules.SparseSequential.forward',
    backend='tensorrt')
def sparse_sequential__forward(self, input: torch.Tensor) -> torch.Tensor:
    from mmcv.ops.sparse_modules import is_spconv_module
    for k, module in self._modules.items():
        if is_spconv_module(module):
            assert isinstance(input, SparseConvTensor)
            self._sparity_dict[k] = input.sparity
            input = module(input)
        else:
            if isinstance(input, SparseConvTensor):
                input.features = module(input.features)
            else:
                input = module(input)
    return input