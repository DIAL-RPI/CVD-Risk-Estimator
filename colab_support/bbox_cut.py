# -*- coding: utf-8 -*-
# @Author  : chq_N
# @Time    : 2020/10/28


import numpy as np

from utils import CT_resize


def calibrate_resizer(image, min_point, max_point, new_size):
    org_space = np.array(image.GetSpacing())
    org_size = max_point - min_point
    new_space = org_space * org_size / new_size
    new_org = min_point * org_space
    return CT_resize(
        image, new_size=new_size.tolist(),
        new_space=new_space.tolist(),
        new_org=new_org.tolist())


def parse_bbox(bbox, bbox_selected, size, space):
    selected = bbox_selected.reshape(-1, 1)
    if selected.sum() <= 10:
        return None, None
    bbox = (bbox * selected).tolist()

    more_slice = int(10 / space[2])
    more_pixel_x = 0
    more_pixel_y = 0
    selected = selected.reshape(-1).tolist()
    last_idx = -1
    for i in range(len(selected)):
        if selected[len(selected) - i - 1] == 1:
            last_idx = len(selected) - i - 1
            break
    if last_idx <= 4:
        return None, None
    # Get mean bbox
    _x = list()
    _y = list()
    _w = list()
    _h = list()
    for i in range(5):
        _x1, _y1, _x2, _y2 = bbox[last_idx - i]
        assert np.sum([_x1, _y1, _x2, _y2]) > 0
        _x.append(np.mean([_x1, _x2]))
        _y.append(np.mean([_y1, _y2]))
        _w.append(_x2 - _x1)
        _h.append(_y2 - _y1)
    c_x = np.mean(_x)
    c_y = np.mean(_y)
    c_w = np.mean(_w)
    c_h = np.mean(_h)
    m_x1 = np.clip(c_x - c_w / 2, 0, size[0])
    m_y1 = np.clip(c_y - c_h / 2, 0, size[1])
    m_x2 = np.clip(c_x + c_w / 2, 0, size[0])
    m_y2 = np.clip(c_y + c_h / 2, 0, size[1])
    for i in range(last_idx - 4, last_idx + 1):
        _x1, _y1, _x2, _y2 = bbox[i]
        bbox[i] = [
            min(_x1, m_x1),
            min(_y1, m_y1),
            max(_x2, m_x2),
            max(_y2, m_y2)]
    for i in range(last_idx + 1, min(last_idx + more_slice, size[2])):
        bbox[i] = [m_x1, m_y1, m_x2, m_y2]
    min_x = size[0]
    min_y = size[1]
    min_z = size[2]
    max_x = max_y = max_z = 0
    for i, _box in enumerate(bbox):
        if np.sum(_box) == 0:
            continue
        z = size[2] - i - 1
        x1, y1, x2, y2 = _box
        x1 = np.clip(int(round(x1) - more_pixel_x), 0, size[0])
        y1 = np.clip(int(round(y1) - more_pixel_y), 0, size[1])
        x2 = np.clip(int(round(x2) + more_pixel_x), 0, size[0])
        y2 = np.clip(int(round(y2) + more_pixel_y), 0, size[1])
        if min_x > x1:
            min_x = x1
        if min_y > y1:
            min_y = y1
        if min_z > z:
            min_z = z
        if max_x < x2:
            max_x = x2
        if max_y < y2:
            max_y = y2
        if max_z < z:
            max_z = z
    if (max_z - min_z + 1) < (30 / space[2]):
        return None, None

    return np.asarray([min_x, min_y, min_z]), np.asarray([max_x, max_y, max_z])


def crop_w_bbox(image, bbox, bbox_selected):
    image.SetOrigin((0, 0, 0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    org_space = np.array(image.GetSpacing())
    try:
        min_point, max_point = parse_bbox(
            bbox, bbox_selected, image.GetSize(), org_space)
    except AssertionError:
        return None
    if min_point is None or max_point is None:
        return None
    return calibrate_resizer(
        image, min_point, max_point,
        np.asarray([128, 128, 128]))
