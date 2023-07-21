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


class IsprsDA(BaseData):
    LABEL_MAP = OrderedDict(
        # Background=-1,
        imp_surf=0,
        building=1,
        low_vege=2,
        tree=3,
        car=4,
        # clutter=5
    )
    COLOR_MAP = OrderedDict(
        # Background=[0, 0, 0],
        imp_surf=[255, 255, 255],
        building=[0, 0, 255],
        low_vege=[0, 255, 255],
        tree=[0, 255, 0],
        car=[255, 255, 0],
        # clutter=[255, 0, 0]
    )
    PALETTE = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()
    SIZE = (512, 512)
    IGNORE_LABEL = -1

    def __init__(self, image_dir, mask_dir, transforms=None, label_type='id'):
        super().__init__(image_dir, mask_dir, transforms, label_type=label_type,
                         offset=-1, ignore_label=self.IGNORE_LABEL, num_class=len(self.LABEL_MAP))
