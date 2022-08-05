'''
Author: haoqiang haoqiang@mindrank.ai
Date: 2022-08-05 02:34:02
LastEditors: haoqiang haoqiang@mindrank.ai
LastEditTime: 2022-08-05 02:35:19
FilePath: /work-home/egnn/dude/utils.py
Description: 

Copyright (c) 2022 by haoqiang haoqiang@mindrank.ai, All Rights Reserved. 
'''
import os
import sys

from icecream import ic
base_dir = os.path.dirname(os.path.dirname(__file__))
ic(base_dir)
sys.path.append(base_dir)

import torch
from torch import nn


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

