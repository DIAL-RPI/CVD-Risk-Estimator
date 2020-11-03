# -*- coding: utf-8 -*-
# @Author  : chq_N
# @Time    : 2019/12/10

import os.path as osp
from copy import deepcopy

import io
import os
import cv2
import numpy as np
import torch
import zipfile
from google.colab import auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def visualize(pic, bbox, caption, selected):
    pic = (pic * 255).astype('uint8')
    draw_caption(pic, bbox, caption)
    x1, y1, x2, y2 = bbox
    if selected:
        cv2.rectangle(pic, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
    else:
        cv2.rectangle(pic, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=1)
    return pic


def calc_iou(a, b):
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_a = (a[2] - a[0]) * (a[3] - a[1])

    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    iw = np.clip(iw, 0, None)
    ih = np.clip(ih, 0, None)

    ua = area_a + area_b - iw * ih
    ua = np.clip(ua, 1e-8, None)
    intersection = iw * ih

    IoU = intersection / ua
    return IoU


def continue_smooth(bbox_selected):
    tmp_idx = [-1] * len(bbox_selected)
    tmp_len = [-1] * len(bbox_selected)
    max_len = 0
    r_idx = -1
    for i, t in enumerate(bbox_selected):
        if t == 0:
            continue
        elif i == 0 or tmp_idx[i - 1] == -1:
            tmp_idx[i] = i
            tmp_len[i] = 1
        else:
            tmp_idx[i] = tmp_idx[i - 1]
            tmp_len[i] = i - tmp_idx[i] + 1
        if max_len < tmp_len[i]:
            max_len = tmp_len[i]
            r_idx = i
    l_idx = tmp_idx[r_idx]
    smoothed_selected = np.zeros(len(bbox_selected))
    smoothed_selected[l_idx:r_idx + 1] = 1
    return smoothed_selected


def load_detector():
    model_name = 'retinanet_heart.pt'
    proj_name = 'detector'
    zip_proj = proj_name + '.zip'
    if not osp.isfile(zip_proj):
        auth.authenticate_user()
        drive_service = build('drive', 'v3')

        print('Downloading the heart detector...')
        file_id = '1TJ4jnarnt98KygPpuqkGWj9eKQE9-aa7'
        request = drive_service.files().get_media(fileId=file_id)
        param = io.BytesIO()
        downloader = MediaIoBaseDownload(param, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()

        param.seek(0)
        with open(zip_proj, 'wb') as f:
            f.write(param.read())
    with zipfile.ZipFile(zip_proj, 'r') as zip_ref:
        zip_ref.extractall('./')
    print('Loading the heart detector...')
    os.chdir(proj_name)
    model = torch.load(model_name)
    os.chdir('../')
    return model


def detector(whole_img):
    retinanet = load_detector()
    print('Detecting heart...')
    retinanet = retinanet.cuda()
    retinanet.eval()

    frame_num = whole_img.shape[0]
    bbox_list = list()
    bbox_selected = list()
    visual_bbox = list()
    for j in range(frame_num - 1, -1, -1):
        pic = np.tile(np.expand_dims(whole_img[j], axis=2), (1, 1, 3))
        torch_pic = torch.Tensor(pic).cuda().float()
        torch_pic = torch_pic.unsqueeze(0).permute(0, 3, 1, 2).contiguous()

        with torch.no_grad():
            scores, classification, transformed_anchors = retinanet(torch_pic)
            scores = scores.data.cpu().numpy()
            transformed_anchors = transformed_anchors.data.cpu().numpy()
            if scores.size == 0:
                scores = np.asarray([0])
                transformed_anchors = np.asarray([[0, 0, 0, 0]])
            bbox_id = np.argmax(scores)
            bbox = np.array(transformed_anchors[bbox_id, :])
            bbox_list.append(bbox)

            score = scores[bbox_id]
            if score > 0.3:
                selected = 1
            elif np.sum(bbox_selected) <= 0:
                selected = 0
            elif bbox_selected[-1] == 1 and calc_iou(
                    bbox_list[-2], bbox_list[-1]) > 0.8 and score > 0.07:
                selected = 1
            else:
                selected = 0
            bbox_selected.append(selected)

            visual_bbox.append(
                visualize(
                    deepcopy(pic), bbox,
                    ': %.3f%%' % (score * 100),
                    selected))

    bbox_list = np.array(bbox_list)
    bbox_selected = continue_smooth(bbox_selected)
    return bbox_list, bbox_selected, visual_bbox
