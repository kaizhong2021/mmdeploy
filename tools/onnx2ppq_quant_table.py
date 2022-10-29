import argparse
import logging
from statistics import mode
from cv2 import _InputArray_STD_ARRAY

from mmcv import Config
from mmdeploy.codebase.base import task

from mmdeploy.utils import Task, get_root_logger, get_task_type, load_config

def get_table(onnx_path: str,
              deploy_cfg: Config,
              model_cfg: Config,
              output_onnx_path: str,
              out_quant_table_path: str,
              image_dir: str = None,
              device: str = 'cuda',
              dataset_type: str = 'val'):
    input_shape = None
    #setup input_shape if existed in 'onnx_configs'
    if 'onnx_config' in deploy_cfg and 'input_shape' in deploy_cfg.onnx_config:
        input_shape = deploy_cfg.onnx_config.input_shape

    #build calibration dataloader. If img dir not specified , use val dataset.
    if image_dir is not None:
        from quant_image_dataset import QuantizationImageDataset
        from torch.utils.data import DataLoader
        dataset = QuantizationImageDataset(
            path=image_dir, deploy_cfg=deploy_cfg, model_cfg=model_cfg)
        dataloader = DataLoader(dataset, batch_size =1)
    else:
        from mmdeploy.apis.utils import build_task_processor
        task_processor = build_task_processor(model_cfg, deploy_cfg, device)
        dataset = task_processor.build_dataloader(model_cfg, dataset_type)
        dataloader = task_processor.build_dataloader(dataset, 1, 1)

    #get an available input shape randomly
    task = get_task_type(deploy_cfg)
    for _, input_data in enumerate(dataloader):
        if task != Task.SUPER_RESOLUTION:

    