from torch.utils.tensorboard import SummaryWriter
import wandb
from enum import Enum
import numpy as np
import torch


class UniWriter(object):
    def __init__(self, project=None,wandb_name='',log_dir=None, suffix=''):

        self.project = project
        if project is not None:
            wandb.init(project=project)
            wandb.run.name = wandb_name
            wandb.run.save()
        self.writer = None
        self.prefix = []
        if log_dir is not None:
            self.writer = SummaryWriter(log_dir=log_dir, filename_suffix=suffix)

    def config(self,config):
        wandb.config.update(config)

    def define_step(self,prefix=None):
        if prefix is not None:
            self.prefix = prefix
            for pf in prefix:
                wandb.define_metric(pf+'/step')
                wandb.define_metric(pf+'/*',step_metric = pf+'/step')

    def add_scalar(self, tag, value, step):
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)
        if self.project is not None:
            dict = {}
            dict[tag] = value
            prefix = tag.split('/')[0]
            if prefix in self.prefix:
                dict[prefix+'/step'] = step
            wandb.log(dict)
    def add_image(self,tag,img_tensor,step,dataformats):
        """img_tensor HWC tensor"""
        if self.writer is not None:
            self.writer.add_image(tag,img_tensor,dataformats=dataformats)
        if self.project is not None:
            img = wandb.Image(img_tensor.cpu().numpy(),mode='RGB',caption=f"image_{tag}_{step}")
            wandb.log({tag:img})
    def add_images(self,tag,img_tensor,step,dataformats):
        """img_tensor NHWC tensor"""
        if self.writer is not None:
            self.writer.add_images(tag,img_tensor,step,dataformats=dataformats)
        if self.project is not None:
            B,H,W,C = img_tensor.shape
            img_concat = img_tensor.permute(1,2,0,3).reshape(H,W*B,C).cpu().numpy()
            img = wandb.Image(img_concat,mode='RGB',caption=f"image_{tag}_{step}")
            wandb.log({tag:img})

