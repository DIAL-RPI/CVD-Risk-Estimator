import SimpleITK as sitk
import numpy as np
from google.colab import files

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

    def load_demo(self):
        pass

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
        print('Resizing the image...')
        old_size = np.asarray(self.org_ct_img.GetSize()).astype('float')
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
        self.detected_npy = sitk.GetArrayFromImage(self.detected_ct_img)
        self.detected_npy = norm(self.detected_npy, -300, 500)
