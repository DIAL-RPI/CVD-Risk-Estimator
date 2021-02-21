# -*- coding: utf-8 -*-
# @Author  : chq_N
# @Time    : 2020/8/29

import io
import os.path as osp
import torch

from googleapiclient.http import MediaIoBaseDownload
from google.colab import auth
from googleapiclient.discovery import build


def load_model(m):
    param_name = 'NLST-Tri2DNet_True_0.0001_16-00700-encoder.ptm'
    if not osp.isfile(param_name):
        print('Please login to download the model parameters.')
        auth.authenticate_user()
        drive_service = build('drive', 'v3')

        print('Downloading the model parameters...')
        file_id = '1H2PFQ_PxXa5ryKmwNivvwGR-hZf5yyGJ'
        request = drive_service.files().get_media(fileId=file_id)
        param = io.BytesIO()
        downloader = MediaIoBaseDownload(param, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()

        param.seek(0)
        with open(param_name, 'wb') as f:
            f.write(param.read())

    print('Loading model parameters...')
    m.encoder.load_state_dict(
        torch.load(param_name))
    print('Model initialized.')
    return m
