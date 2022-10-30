# Copyright (c) OpenMMLab. All rights reserved.
from matplotlib import backend_bases
from mmdeploy.backend.tensorrt import is_available, is_custom_ops_available
from ..core import PIPELINE_MANAGER

__all__ = ['is_available', 'is_custom_ops_available']

if is_available():
    from mmdeploy.backend.tensorrt import from_onnx as _from_onnx
    from mmdeploy.backend.tensorrt import load, save
    from mmdeploy.backend.tensorrt.quant import get_quant_model_file
    from_onnx = PIPELINE_MANAGER.register_pipeline()(_from_onnx)
    __all__ += ['from_onnx', 'save', 'load','get_quant_model_file']
    try:
        from mmdeploy.backend.tensorrt.onnx2tensorrt import \
            onnx2tensorrt as _onnx2tensorrt

        onnx2tensorrt = PIPELINE_MANAGER.register_pipeline()(_onnx2tensorrt)
        __all__ += ['onnx2tensorrt']
    except Exception:
        pass
