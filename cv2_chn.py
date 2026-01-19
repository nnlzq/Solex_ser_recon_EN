# -*- coding: utf-8 -*-
# 提高cv2.imread/imwrite对中文目录和文件名的兼容性

import cv2
import numpy as np

def imread(path,arg):
    """读取中文路径图片"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint16), -1,arg)

def imwrite(path, img, arg):
    """保存图片到中文路径"""
    ext = path.split('.')[-1].lower()
    result, encoded_img = cv2.imencode(f'.{ext}', img, arg)
    if result:
        encoded_img.tofile(path)
        return True
    return False

