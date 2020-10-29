# -*- coding: utf-8 -*-
# @Author  : chq_N
# @Time    : 2020/10/28


import SimpleITK as sitk


def CT_resize(image, new_size=None, new_space=None, new_direction=None, new_org=None):
    if new_size is None:
        new_size = image.GetSize()
    if new_space is None:
        new_space = image.GetSpacing()
    if new_direction is None:
        new_direction = image.GetDirection()
    if new_org is None:
        new_org = image.GetOrigin()
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection(new_direction)
    resampler.SetOutputSpacing(new_space)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(new_org)
    resampler.SetInterpolator(sitk.sitkGaussian)
    resampler.SetDefaultPixelValue(
        sitk.GetArrayFromImage(image).min().astype('float'))
    return resampler.Execute(image)


def norm(input_array, norm_down, norm_up):
    input_array = input_array.astype('float32')
    normed_array = (input_array - norm_down) / (norm_up - norm_down)
    normed_array[normed_array > 1] = 1
    normed_array[normed_array < 0] = 0
    return normed_array


def visualize_data(npy_img):
    print(npy_img)