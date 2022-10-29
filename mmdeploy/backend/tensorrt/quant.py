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

#def tensorrt2int8()
