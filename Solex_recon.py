# -*- coding: utf-8 -*-
"""
@author: Andrew Smith
contributors: Valerie Desnoux, Jean-Francois Pittet, Jean-Baptiste Butet, Pascal Berteau, Matt Considine
Version 24 September 2023

------------------------------------------------------------------------
Reconstruction of an image from the deviations between the minimum of the line and a reference line
-------------------------------------------------------------------------

"""

from solex_util import *
from video_reader import *
from ellipse_to_circle import ellipse_to_circle, correct_image
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
import FreeSimpleGUI as sg # for progress bar
from scipy.ndimage import gaussian_filter1d
import cv2
import numpy as np
from astropy.io import fits  

'''
process files: call solex_read and solex_proc to process a list of files with specified options
input: tasks: list of tuples (file, option)
'''

def solex_do_work(tasks, flag_command_line = False):
    multi = True
    if not multi:
        print("WARNING: multithreading is off")
    with Pool(4) as p:
        results = []
        for i, (file, options) in enumerate(tasks):
            print('file %s is processing'%file)
            if len(tasks) > 1 and not flag_command_line:
                sg.one_line_progress_meter('Progress Bar', i, len(tasks), '','Reading file...')
            disk_list, backup_bounds, hdr = solex_read(file, options)
            if multi:
                result = p.apply_async(solex_process, args = (options, disk_list, backup_bounds, hdr)) # TODO: prints won't be visible inside new thread, can this be fixed?
                results.append(result)
            else:
                solex_process(options, disk_list, backup_bounds, hdr)
        [result.get() for result in results]
        if len(tasks) > 1 and not flag_command_line:
            sg.one_line_progress_meter('Progress Bar', len(tasks), len(tasks), '','Done.')

'''
read a solex file and return a list of numpy arrays representing the raw result
'''
def solex_read(file, options):
    basefich0 = os.path.splitext(file)[0] # file name without extension
    options['basefich0'] = basefich0
    clearlog(basefich0 + '_log.txt', options)
    logme(basefich0 + '_log.txt', options, 'Pixel shift : ' + str(options['shift']))
    options['shift_requested'] = options['shift']
    options['shift'] = list(dict.fromkeys([options['ellipse_fit_shift'], 0] + options['shift']))  # options['ellipse_fit_shift'], 0 are "fake", but if they are requested, then don't double count
    rdr = video_reader(file)
    hdr = make_header(rdr)
    ih = rdr.ih
    iw = rdr.iw

    mean_img, fit, backup_y1, backup_y2 = compute_mean_return_fit(video_reader(file), options, hdr, iw, ih, basefich0)

    disk_list, ih, iw, FrameCount = read_video_improved(video_reader(file), fit, options)

    hdr['NAXIS1'] = iw  # note: slightly dodgy, new width for subsequent fits file



    # sauve fichier disque reconstruit

    if options['flag_display']:
        cv2.destroyAllWindows()

    for i in range(len(disk_list)):
        if options['flip_x']:
            disk_list[i] = np.flip(disk_list[i], axis = 1)
        basefich = basefich0 + '_shift=' + str(options['shift'][i])
        flag_requested = options['shift'][i] in options['shift_requested']

        if options['save_fit'] and flag_requested:
            DiskHDU = fits.PrimaryHDU(disk_list[i], header=hdr)
            DiskHDU.writeto(output_path(basefich + '_raw.fits', options), overwrite='True')
    return disk_list, (backup_y1, backup_y2), hdr

'''
process the raw disks: circularise, detransversalium, crop, and adjust contrast

inputs: disk_list : list of images as np arrays
backup_bounds: tuple of numbers for disk upper and lower bounds (backup for case of no ellipse-fit)
hdr: an hdr header for fits files

'''
def solex_process(options, disk_list, backup_bounds, hdr):
    basefich0 = options['basefich0']
    if options['transversalium']:
        logme(basefich0 + '_log.txt', options, 'Transversalium correction : ' + str(options['trans_strength']))
    else:
        logme(basefich0 + '_log.txt', options, 'Transversalium disabled')
    logme(basefich0 + '_log.txt', options, 'Mirror X : ' + str(options['flip_x']))
    logme(basefich0 + '_log.txt', options, 'Post-rotation : ' + str(options['img_rotate']) + ' degrees')
    logme(basefich0 + '_log.txt', options, f'Protus adjustment : {options["delta_radius"]}')
    logme(basefich0 + '_log.txt', options, f'de-vignette : {options["de-vignette"]}')
    borders = [0,0,0,0]
    cercle0 = (-1, -1, -1)
    for i in range(len(disk_list)):
        flag_requested = options['shift'][i] in options['shift_requested']
        basefich = basefich0 + '_shift=' + str(options['shift'][i])
        """
        We now apply ellipse_fit to apply the geometric correction

        """
        # disk_list[0] is always shift = 10, for more contrast for ellipse fit
        if options['ratio_fixe'] is None and options['slant_fix'] is None:
            frame_circularized, cercle0, options['ratio_fixe'], phi, borders = ellipse_to_circle(
                disk_list[i], options, basefich)
            # in options angles are stored as degrees (slightly annoyingly)
            options['slant_fix'] = math.degrees(phi)

        else:
            ratio = options['ratio_fixe'] if not options['ratio_fixe'] is None else 1.0
            phi = math.radians(options['slant_fix']) if not options['slant_fix'] is None else 0.0
            if flag_requested:
                frame_circularized = correct_image(disk_list[i] / 65536, phi, ratio, np.array([-1.0, -1.0]), -1.0, options, print_log=i == 0)[0]  # Note that we assume 16-bit
                if options['de-vignette']:
                    if cercle0 == (-1, -1, -1):
                        print("WARNING: cannot de-vignette without ellipse fit")
                    else:
                        frame_circularized = removeVignette(frame_circularized, cercle0)
        if not flag_requested:
            continue # skip processing if shift is not desired

        single_image_process(frame_circularized, hdr, options, cercle0, borders, basefich, backup_bounds)
        write_complete(basefich0 + '_log.txt', options)




def single_image_process(frame_circularized, hdr, options, cercle0, borders, basefich, backup_bounds):
    if options['save_fit']:
        DiskHDU = fits.PrimaryHDU(frame_circularized, header=hdr)
        DiskHDU.writeto(output_path(basefich + '_circular.fits', options), overwrite='True')

    # 横向畸变校正
    if options['transversalium']:
        if not cercle0 == (-1, -1, -1):
            detransversaliumed = correct_transversalium2(frame_circularized, cercle0, borders, options, 0, basefich)
        else:
            detransversaliumed = correct_transversalium2(
                frame_circularized, 
                (0,0,99999), 
                [0, backup_bounds[0]+20, frame_circularized.shape[1] -1, backup_bounds[1]-20], 
                options, 0, basefich
            )
    else:
        detransversaliumed = frame_circularized

    if options['save_fit'] and options['transversalium']:
        DiskHDU = fits.PrimaryHDU(detransversaliumed, header=hdr)
        DiskHDU.writeto(output_path(basefich + '_detransversaliumed.fits', options), overwrite='True')

    # ========== 核心修复：稳定的圆心定位 + 坐标体系修正 + 保守居中 ==========
    def get_stable_circle_center(img, cercle0):
        """
        稳定的圆心定位策略：
        1. 优先使用椭圆拟合的圆心（cercle0），这是天文图像的可靠来源
        2. 若拟合失败，用图像灰度重心（比霍夫圆更稳定）
        3. 最后兜底用图像几何中心
        """
        h, w = img.shape
        
        # 策略1：优先使用椭圆拟合的圆心（原逻辑，最可靠）
        if not cercle0 == (-1, -1, -1):
            cx, cy = int(cercle0[0]), int(cercle0[1])
            #print(f"使用椭圆拟合圆心：({cx}, {cy})")
            return cx, cy
        
        # 策略2：灰度重心（基于亮度分布，比霍夫圆稳定）
        # 归一化图像
        img_norm = (img - np.min(img)) / (np.max(img) - np.min(img))
        # 计算水平/垂直方向的灰度重心
        x_coords = np.arange(w)
        y_coords = np.arange(h)
        # 水平重心：各列亮度加权平均
        cx = np.sum(x_coords * np.sum(img_norm, axis=0)) / np.sum(img_norm)
        # 垂直重心：各行亮度加权平均
        cy = np.sum(y_coords * np.sum(img_norm, axis=1)) / np.sum(img_norm)
        cx, cy = int(round(cx)), int(round(cy))
        
        if not np.isnan(cx) and not np.isnan(cy):
            #print(f"使用灰度重心圆心：({cx}, {cy})")
            return cx, cy
        
        # 策略3：兜底用几何中心
        #print("使用图像几何中心：({w//2}, {h//2})")
        return w//2, h//2

    def center_image_around_point(img, target_center, target_size):
        """
        保守的居中逻辑：将指定圆心移到图像正中心，而非裁剪
        （避免裁剪导致的偏移，直接平移/填充）
        """
        h, w = img.shape
        target_w, target_h = target_size
        # 创建背景画布（填充原图像背景色：边缘像素的均值，更自然）
        bg_color = np.mean(img[0:10, 0:10])  # 取左上角10x10区域的均值作为背景
        new_img = np.full((target_h, target_w), bg_color, dtype=img.dtype)
        
        # 计算平移偏移量：让target_center对齐新图像中心
        new_cx, new_cy = target_w // 2, target_h // 2
        dx = new_cx - target_center[0]  # 水平偏移
        dy = new_cy - target_center[1]  # 垂直偏移
        
        # 计算原图像在新画布中的位置（避免越界）
        # 原图像的显示区域
        src_x_start = max(0, -dx)
        src_x_end = min(w, target_w - dx)
        src_y_start = max(0, -dy)
        src_y_end = min(h, target_h - dy)
        
        # 新画布的粘贴区域
        dst_x_start = max(0, dx)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)
        dst_y_start = max(0, dy)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        
        # 粘贴图像（核心：仅平移，不裁剪，避免圆心偏移）
        new_img[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = img[src_y_start:src_y_end, src_x_start:src_x_end]
        
        return new_img

    cercle = cercle0
    if not options['fixed_width'] is None or options['crop_width_square']:
        h, w = detransversaliumed.shape
        # 确定目标尺寸（保持原逻辑，但改为平移居中而非裁剪）
        target_w = h if options['fixed_width'] is None else options['fixed_width']
        target_h = target_w if options['crop_width_square'] else h
        target_size = (target_w, target_h)
        
        # 第一步：获取稳定的圆心（优先椭圆拟合，避免错误检测）
        cx, cy = get_stable_circle_center(detransversaliumed, cercle0)
        
        # 第二步：保守居中：将圆心平移到图像中心（不裁剪，仅填充）
        detransversaliumed = center_image_around_point(detransversaliumed, (cx, cy), target_size)
        
        # 更新圆心为新图像的几何中心（严格居中）
        if not cercle == (-1, -1, -1):
            cercle = (target_w // 2, target_h // 2, cercle[2])

    # 最终图像处理
    return image_process(detransversaliumed, cercle, options, hdr, basefich)
    
    