import os.path as osp
from subprocess import call
from typing import List

import mmcv

from .init_plugins import get_onnx2tensorrt_path

def get_quant_model_file(onnx_path: str, work_dir: str) -> List[str]:
    """Returns the path to quant onnx and table with export results.

    Args:
        onnx_path (str): The path to the int8 onnx model.
        work_dir(str): The path to the directory for saving the results.

    Returns:
        List[str]: The path to the files where the export result will be 
            located.
    """

    mmcv.mkdir_or_exist(osp.abspath(work_dir))
    base_name = osp.splitext(osp.split(onnx_path)[1])[0]
    quant_onnx = osp.join(work_dir, base_name + '_quant.onnx')
    quant_engine = osp.join(work_dir, base_name + '_int8.engine')
    return [quant_onnx,quant_engine]

def tensorrt2int8(engine: str, int8_engine: str):
    """Convert tensorrt float model to quantized model.

    The inputs of tensorrt include float model and weight file. We need to use
    a executable program to convert the float model to int8 model with
    calibration table.

    Example:
        >>> from mmdeploy.backend.tensorrt.quant import tensorrt2int8
        >>> engine = 'work_dir/end2end.engine'
        >>> int8_engine = 'work_dir/end2end_int8.engine'
        >>> ncnn2int8(engine, table, int8_engine)

    Args:
        param (str): The path of ncnn float model graph.
        bin (str): The path of ncnn float weight model weight.
        table (str): The path of ncnn calibration table.
        int8_param (str):  The path of ncnn low bit model graph.
        int8_bin (str): The path of ncnn low bit weight model weight.
    """

    tensorrt2int8 = get_onnx2tensorrt_path()

    call([tensorrt2int8, engine, int8_engine ])