"""
@Project :
@File    :
@IDE     : PyCharm
@Author  : Wang Liu
@Date    :
@e-mail  : 1183862787@qq.com
"""
import logging
import numpy as np
from module.datasets.basedata import BaseData
from collections import OrderedDict


logger = logging.getLogger(__name__)

class LoveDA(BaseData):

    LABEL_MAP = OrderedDict(
        Background=0,
        Building=1,
        Road=2,
        Water=3,
        Barren=4,
        Forest=5,
        Agricultural=6
    )

    COLOR_MAP = OrderedDict(
        Background=(255, 255, 255),
        Building=(255, 0, 0),
        Road=(255, 255, 0),
        Water=(0, 0, 255),
        Barren=(159, 129, 183),
        Forest=(0, 255, 0),
        Agricultural=(255, 195, 128),
    )

    PALETTE = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()
    SIZE=(1024, 1024)
    IGNORE_LABEL=-1

    def __init__(self, image_dir, mask_dir, transforms=None, label_type='id'):
        super().__init__(image_dir, mask_dir, transforms, label_type, offset=-1, ignore_label=LoveDA.IGNORE_LABEL)
