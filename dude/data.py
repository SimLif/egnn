'''
Author: haoqiang haoqiang@mindrank.ai
Date: 2022-08-05 04:02:19
LastEditors: haoqiang haoqiang@mindrank.ai
LastEditTime: 2022-08-05 06:14:18
FilePath: /work-home/egnn/dude/data.py
Description: 

Copyright (c) 2022 by haoqiang haoqiang@mindrank.ai, All Rights Reserved. 
'''
import os
import sys

from icecream import ic
base_dir = os.path.dirname(os.path.dirname(__file__))
ic(base_dir)
sys.path.append(base_dir)

class DataUp:
    pass