# Copyright (c) OpenMMLab. All rights reserved.
import ctypes
import os
import shutil
from mmdeploy.backend.tensorrt.onnx2tensorrt import onnx2tensorrt

from mmdeploy.utils import get_file_path, get_root_logger


def get_ops_path() -> str:
    """Get path of the TensorRT plugin library.

    Returns:
        str: A path of the TensorRT plugin library.
    """
    candidates = [
        '../../lib/libmmdeploy_tensorrt_ops.so',
        '../../lib/mmdeploy_tensorrt_ops.dll',
        '../../../build/lib/libmmdeploy_tensorrt_ops.so',
        '../../../build/bin/*/mmdeploy_tensorrt_ops.dll'
    ]
    return get_file_path(os.path.dirname(__file__), candidates)


def load_tensorrt_plugin() -> bool:
    """Load TensorRT plugins library.

    Returns:
        bool: True if TensorRT plugin library is successfully loaded.
    """
    lib_path = get_ops_path()
    success = False
    logger = get_root_logger()
    if os.path.exists(lib_path):
        ctypes.CDLL(lib_path)
        logger.info(f'Successfully loaded tensorrt plugins from {lib_path}')
        success = True
    else:
        logger.warning(f'Could not load the library of tensorrt plugins. \
            Because the file does not exist: {lib_path}')
    return success

def get_onnx2tensorrt_path() -> str:
    """Get mmdeploy_onnx2tensorrt path.

    Returns:
    str: A path of mmdeploy_tensorrt tool.
    """
    candidates = ['./mmdeploy_onnx2tensorrt', './mmdeploy_onnx2tensorrt.exe']
    onnx2tensorrt_path = get_file_path(os.path.dirname(__file__), candidates)

    if onnx2tensorrt_path is None or not os.path.exists(onnx2tensorrt_path):
        onnx2tensorrt_path = get_file_path('', candidates)

    if onnx2tensorrt_path is None or not os.path.exists(onnx2tensorrt_path):
        onnx2tensorrt_path = shutil.which('mmdeploy_onnx2tensorrt')
        onnx2tensorrt_path = '' if onnx2tensorrt_path is None else onnx2tensorrt_path

    return onnx2tensorrt_path

def get_tensorrt2int8_path() -> str:
    """Get onnx2int8 path.

    Returns:
        str: A path of tensorrt2int8 tools.
    """
    tensorrt2int8_path = shutil.which('tensorrt2int8')
    if tensorrt2int8_path is None:
        raise Exception(
            'Cannot find tensorrt2int8, try `export PATH=/path/to/tensorrt2int8`'
    )
    return tensorrt2int8_path