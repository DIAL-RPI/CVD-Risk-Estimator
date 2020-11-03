import io
import SimpleITK as sitk
import numpy as np
import os.path as osp

from google.colab import files
from googleapiclient.http import MediaIoBaseDownload
from google.colab import auth
from googleapiclient.discovery import build

from .bbox_cut import crop_w_bbox
from .heart_detect import detector
from .utils import norm, CT_resize


class Image:
    CT_AXIAL_SIZE = 512

    def __init__(self):
        self.org_ct_img = None
        self.org_npy = None
        self.bbox = None
        self.bbox_selected = None
        self.visual_bbox = None
        self.detected_ct_img = None
        self.detected_npy = None

    def load_demo(self, demo_id):
        demo_id = str(demo_id)
        file_id_dict = {
            '1': '1DlW6NYvJYkZ_wMw4tgk_VfY2M-Yu1A5l',
            '2': '1IYul8RXENFTncHz0xYzTgney-n_g5EZf',
            '3': '1o-NiPKDUkOqiKO7wyY4DyRUER-teCpnw',
            '4': '1NVqEd3qDpJ1W5IjVeBg8YbO07PJ8ThMk',
            '5': '104pDbWRt3zd33778qmOrNHLErt89CY2F',}
        if demo_id not in file_id_dict:
            print('Sorry we do not have a demo with ID', demo_id)
            return
        file_id = file_id_dict[demo_id]
        file_save_name = demo_id + '.nii'
        if not osp.isfile(file_save_name):
            auth.authenticate_user()
            drive_service = build('drive', 'v3')

            print('Downloading Demo %s...' % demo_id)
            request = drive_service.files().get_media(fileId=file_id)
            demo = io.BytesIO()
            downloader = MediaIoBaseDownload(demo, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()

            demo.seek(0)
            with open(file_save_name, 'wb') as f:
                f.write(demo.read())
        self.org_ct_img = sitk.ReadImage(file_save_name)

    def upload_heart_region_nifti(self):
        uploaded = files.upload()
        file_name = uploaded.keys()[0]
        self.detected_ct_img = sitk.ReadImage(file_name)
        old_size = np.asarray(self.detected_ct_img.GetSize()).astype('float')
        new_size = np.asarray([128, 128, 128]).astype('float')
        old_space = np.asarray(self.detected_ct_img.GetSpacing()).astype('float')
        new_space = old_space * old_size / new_size
        self.detected_ct_img = CT_resize(
            self.detected_ct_img,
            new_size=new_size.astype('int').tolist(),
            new_space=new_space.tolist())
        self.detected_npy = sitk.GetArrayFromImage(self.detected_ct_img)
        self.detected_npy = norm(self.detected_npy, -300, 500)

    def upload_nifti(self):
        uploaded = files.upload()
        file_name = list(uploaded.keys())[0]
        self.org_ct_img = sitk.ReadImage(file_name)

    def detect_heart(self):
        # Resize org ct
        old_size = np.asarray(self.org_ct_img.GetSize()).astype('float')
        if min(old_size[0], old_size[1]) < 480 or max(old_size[0], old_size[1]) > 550:
            print('Resizing the image...')
            new_size = np.asarray([
                Image.CT_AXIAL_SIZE, Image.CT_AXIAL_SIZE, old_size[-1]]
            ).astype('float')
            old_space = np.asarray(self.org_ct_img.GetSpacing()).astype('float')
            new_space = old_space * old_size / new_size
            self.org_ct_img = CT_resize(
                self.org_ct_img,
                new_size=new_size.astype('int').tolist(),
                new_space=new_space.tolist())
        self.org_npy = sitk.GetArrayFromImage(self.org_ct_img)
        self.org_npy = norm(self.org_npy, -500, 500)
        print('Detecting heart...')
        self.bbox, self.bbox_selected, self.visual_bbox = detector(self.org_npy)
        self.detected_ct_img = crop_w_bbox(
            self.org_ct_img, self.bbox, self.bbox_selected)
        if self.detected_ct_img is None:
            print('Fail to detect heart in the image. '
                  'Please manually crop the heart region.')
            return
        self.detected_npy = sitk.GetArrayFromImage(self.detected_ct_img)
        self.detected_npy = norm(self.detected_npy, -300, 500)
