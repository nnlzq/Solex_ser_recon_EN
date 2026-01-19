# -*- coding: utf-8 -*-
"""
@author: Andrew Smith
based on code by Valerie Desnoux
contributors: Jean-Francois Pittet, Jean-Baptiste Butet, Pascal Berteau, Matt Considine
Version 24 September 2023

--------------------------------------------------------------
Front end of spectroheliograph processing of SER and AVI files
- interface able to select one or more files
- call to the solex_recon module which processes the sequence and generates the PNG and FITS files
- offers with openCV a display of the resultant image
- wavelength selection with the pixel shift function, including multiple wavelengths and a range of wavelengths
- geometric correction with a fixed Y/X ratio
- if Y/X is blank, then this will be calculated automatically
--------------------------------------------------------------

"""
import math
import numpy as np

import os
import sys
import Solex_recon
import UI_handler
import CLI_handler
from astropy.io import fits
import cProfile
import FreeSimpleGUI as sg
import traceback
import cv2
import json
import time
from multiprocessing import freeze_support
import glob
import solex_util
import video_reader
import datetime

import multiprocessing

multiprc_num=(os.cpu_count() // 2)

serfiles = []

options = {
    'language':'English',           #
    'shift':[0],                    # argument: w
    'flag_display':False,           # argument: d
    'ratio_fixe' : None,            # argument: x
    'slant_fix' : None ,            #
    'save_fit' : False,             # argument: f
    'clahe_only' : False,           # argument: c
    'protus_only': False,
    'disk_display' : True,          # argument: p
    'delta_radius' : 0,             #
    'crop_width_square' : False,    # argument: s
    'transversalium' : True,        # argument: t
    'stubborn_transversalium': False,
    'trans_strength': 301,          #
    'img_rotate': 0,                #
    'flip_x': False,                # argument: m
    'workDir': '',                  #
    'fixed_width': None,            # argument: r
    'output_dir': '',               #
    'input_dir': '',                #
    'specDir': '',                  # for spectral analyser
    'selected_mode': 'File input mode',
    'continuous_detect_mode': False,#
    'dispersion':0.05,              # for spectral analyser
    'ellipse_fit_shift':10,         # secret parameter for ellipse fit
    'de-vignette':False             # remove vigenette
}


'''
open config.txt and read parameters
return parameters from file, or default if file not found or invalid
'''
def read_ini():
    # check for config.txt file for working directory
    print('loading config file...')

    try:
        mydir_ini=os.path.join(os.path.dirname(sys.argv[0]),'SHG_config.txt')
        with open(mydir_ini, 'r', encoding="utf-8") as fp:
            global options
            options.update(json.load(fp)) # if config has missing entries keep default
    except Exception:
        print('note: error reading config file - using default parameters')


def write_ini():
    try:
        print('saving config file ...')
        mydir_ini = os.path.join(os.path.dirname(sys.argv[0]),'SHG_config.txt')
        with open(mydir_ini, 'w', encoding="utf-8") as fp:
            json.dump(options, fp, sort_keys=True, indent=4)
    except Exception:
        traceback.print_exc()
        print('ERROR: failed to write config file: ' + mydir_ini)

# 单个文件预处理函数（顶层函数，支持多进程序列化调用）
def _check_single_serfile(args):
    """
    单个.ser文件的校验与预处理（供进程池调用）
    :param args: 元组参数 (serfile, options_copy)，适配进程池imap方法
    :return: 处理结果：有效文件返回(serfile, options_copy)，无效返回None
    """
    serfile, options_copy = args
    try:
        # 1. 空文件名校验
        if serfile == '':
            print(f"ERROR filename empty: {serfile}")
            return None
        
        # 2. 文件名基础格式校验
        base = os.path.basename(serfile)
        if base == '':
            print(f'filename ERROR : {serfile}')
            return None
        
        # 3. 尝试打开文件，校验文件可访问性
        try:
            with open(serfile, "rb") as f:  # 使用with语句，自动关闭文件，更安全
                pass
        except Exception as e:
            traceback.print_exc()
            print(f'ERROR opening file : {serfile}')
            return None
        
        # 4. 监测文件大小是否稳定（避免文件正在写入）
        fsize = os.path.getsize(serfile)
        while fsize > 0:
            time.sleep(3)
            fsize2 = os.path.getsize(serfile)
            if fsize == fsize2:
                break
            else:
                fsize = fsize2
        
        # 5. 有效文件，返回结果（保持原逻辑的options.copy()）
        return (serfile, options_copy)
    
    except Exception as e:
        traceback.print_exc()
        print(f'Unexpected error processing file : {serfile}')
        return None

# 改造后的multiprc_num进程并行版precheck_files（保持原函数签名不变）
def precheck_files(serfiles, options):
    # 原逻辑：根据文件数量设置tempo参数（主进程执行，仅执行一次）
    if len(serfiles) == 1:
        options['tempo'] = 30000  # 单个文件，延长展示时间
    else:
        options['tempo'] = 5000   # 多个文件，缩短展示时间

    # 步骤1：预处理任务参数，为每个文件准备独立的options副本
    task_args = []
    for serfile in serfiles:
        print(f"Submitting file for precheck: {serfile}")
        # 每个任务携带独立的options副本，避免多进程共享修改
        task_args.append((serfile, options.copy()))
    
    # 步骤2：创建固定multiprc_num个进程的进程池，并行处理所有文件
    good_tasks = []
    with multiprocessing.Pool(processes=multiprc_num) as pool:
        # 用imap_unordered异步获取处理结果，效率高于map（支持实时返回完成结果）
        for result in pool.imap_unordered(_check_single_serfile, task_args):
            if result is not None:  # 筛选有效结果，忽略无效文件
                good_tasks.append(result)
    
    # 步骤3：保持原逻辑的ini文件写入（主进程统一处理，避免多进程并发写入冲突）
    if good_tasks:
        # 若存在有效任务，提取第一个任务的路径设置workDir（保持原逻辑）
        first_serfile, first_options = good_tasks[0]
        if first_options['selected_mode'] == 'File input mode':
            first_options['workDir'] = os.path.dirname(first_serfile) + "/"
        write_ini()
    else:
        # 无有效任务，也执行ini写入（保持原逻辑）
        write_ini()
    
    # 步骤4：返回有效任务列表（格式与原函数一致）
    return good_tasks

def single_task_processor(task):
    """
    单个任务处理函数（供进程池调用，顶层可序列化函数）
    :param task: 单个任务元组，格式为(file, options)
    :return: 任务处理结果（文件名称、状态、详情）
    """
    file, options = task
    try:
        # 调用核心业务逻辑，与原流程保持一致
        disk_list, backup_bounds, hdr = Solex_recon.solex_read(file, options)
        result = Solex_recon.solex_process(options, disk_list, backup_bounds, hdr)
        return (file, "SUCCESS", result)
    except Exception as e:
        # 捕获单个任务内部异常，不影响其他任务执行
        return (file, "FAILED", str(e))

# 保持原函数参数个数不变：仅files, options, flag_command_line
def handle_files(files, options, flag_command_line=False):
    """
    多进程版本文件处理函数（固定multiprc_num个进程，保持原参数个数不变）
    :param files: 待处理文件列表
    :param options: 处理配置参数
    :param flag_command_line: 是否命令行模式（不显示进度条）
    """
    # 记录开始时间
    time_start = datetime.datetime.now()

    # 步骤1：预处理文件，筛选有效任务
    good_tasks = precheck_files(files, options)
    task_count = len(good_tasks)
    #print(f"文件预处理完成，筛选出 {task_count} 个有效任务待处理")
    
    if task_count == 0:
        #print("无有效任务，直接退出")
        return

    # 步骤2：初始化进度条（非命令行模式）
    completed_count = 0
    if task_count > 1 and not flag_command_line:
        sg.one_line_progress_meter(
            f'{multiprc_num}-Process Task Progress',
            0,
            task_count,
            'Total Tasks',
            f'Initializing {multiprc_num} processes...'
        )

    # 步骤3：创建固定multiprc_num个进程的进程池，并行处理任务
    results = []
    try:
        # 固定进程数=multiprc_num，启用maxtasksperchild避免内存泄露
        with multiprocessing.Pool(processes=multiprc_num, maxtasksperchild=1) as pool:
            # 使用imap_unordered实时获取任务结果，支持进度条实时更新
            for task_result in pool.imap_unordered(single_task_processor, good_tasks):
                results.append(task_result)
                completed_count += 1

                # 步骤4：更新进度条（反映4进程实际处理进度）
                if task_count > 1 and not flag_command_line:
                    sg.one_line_progress_meter(
                        f'{multiprc_num}-Process Task Progress',
                        completed_count,
                        task_count,
                        'Total Tasks',
                        f'Completed {completed_count}/{task_count} tasks'
                    )

        # 步骤5：打印任务处理汇总结果
        #print("\n===== 任务处理汇总 =====")
        success_count = 0
        for file, status, detail in results:
            if status == "SUCCESS":
                success_count += 1
                #print(f"文件 {file}：处理成功 - {detail}")
            else:
                #print(f"文件 {file}：处理失败 - {detail}")
                pass
        #print(f"===== 汇总结束 =====")
        #print(f"任务总数量：{task_count}，成功数量：{success_count}，失败数量：{task_count - success_count}")

    except Exception as e:  # 捕获进程池框架相关异常
        print(f'ERROR ENCOUNTERED IN {multiprc_num}-PROCESS FRAMEWORK')
        traceback.print_exc()
        
        # 关闭OpenCV窗口（按需保留，消除原TODO疑问）
        cv2.destroyAllWindows()
        
        # 非命令行模式弹出错误弹窗
        if not flag_command_line:
            error_msg = f"{multiprc_num}-Process Framework Error:\n\n{traceback.format_exc()}"
            sg.popup_ok(error_msg, title="Multi-Process Error", font=("Arial", 10))

    finally:
        # 步骤6：格式化输出耗时统计
        time_end = datetime.datetime.now()
        time_cost = time_end - time_start
        
        total_seconds = int(time_cost.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        microseconds = time_cost.microseconds // 1000  # 保留毫秒级精度
        
        if hours > 0:
            time_cost_str = f"{hours}h {minutes}m {seconds}.{microseconds:03d}s"
        elif minutes > 0:
            time_cost_str = f"{minutes}m {seconds}.{microseconds:03d}s"
        else:
            time_cost_str = f"{seconds}.{microseconds:03d}s"
        
        print(f"\n----> Time cost ({multiprc_num}-Process Mode):  {time_cost_str}\n")
        
        # 关闭最终进度条（非命令行模式）
        if task_count > 1 and not flag_command_line:
            sg.one_line_progress_meter(f'{multiprc_num}-Process Task Progress', task_count, task_count, 'Total Tasks', 'Done.')

def is_openable(file):
    try:
        f=open(file, "rb")
        f.close()
        rdr = video_reader.video_reader(file)
        return rdr.FrameCount > 0
    except:
        return False

def handle_folder(options):
    if not options['continuous_detect_mode']:
        files_todo = glob.glob(os.path.join(options['input_dir'], '*.ser')) + glob.glob(os.path.join(options['input_dir'], '*.avi'))
        print(f'number of files todo: {len(files_todo)}')
        handle_files(files_todo, options)
        return

    files_processed = set()
    layout = [
        [sg.Text('Auto processing of SHG video files', font='Any 12', key='Auto processing of SHG video files'), sg.Push(), sg.Button('Stop')],
        [sg.Text(f'Number of files processed: {len(files_processed)}', key='auto_info'), sg.Push(), sg.Text('Looking for files ...', key='status_info')],
        [sg.Image(UI_handler.resource_path(os.path.join('language_data', 'Black.png')), size=(600, 600), key='_prev_img')],
        [sg.Text('Last: none', key='last')],
    ]
    #window = sg.Window('Continuous processing mode', layout, keep_on_top=True)
    window = sg.Window( options['input_dir'].split("/")[-1], layout, keep_on_top=False)
    window.finalize()
    stop=False


    window.perform_long_operation(lambda : time.sleep(0.01), '-END SLEEP-') # dummy function to get started
    prev=None
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        if event == '-END KEY-':
            window['status_info'].update('Looking for files ...')
            window['auto_info'].update(f'Number of files processed: {len(files_processed)}')
            if stop:
                window.close()
                break
            time.sleep(0.1)
            if not prev is None:
                window['_prev_img'].update(data=UI_handler.get_img_data(prev, maxsize=(600,600), first=True))
                window['last'].update('Last: ' + prev)
            window.perform_long_operation(lambda : time.sleep(1), '-END SLEEP-')

        if event == '-END SLEEP-':
            files_todo = glob.glob(os.path.join(options['input_dir'], '*.ser')) + glob.glob(os.path.join(options['input_dir'], '*.avi'))
            files_todo = [x for x in files_todo if not x in files_processed and os.access(x, os.R_OK) and is_openable(x)]
            files_todo = files_todo[:min(1, len(files_todo))] # maximum batch size 1
            if files_todo:
                window['status_info'].update(f'About to process {len(files_todo)} file')
                prev=files_todo[-1]
                prev=os.path.join(solex_util.output_path(os.path.splitext(prev)[0] + f'_shift={options["shift"][-1]}_clahe.png', options)).replace('\\', "/")
                print('the image file:' + str(prev))
                window.perform_long_operation(lambda : handle_files(files_todo, options, True), '-END KEY-')
            else:
                window['status_info'].update('Looking for files ...')
                window.perform_long_operation(lambda : time.sleep(1), '-END KEY-')
            files_processed.update(files_todo)

        if event == 'Stop':
            stop=True
            window['status_info'].update(f'WILL STOP AFTER PROCESSING CURRENT BATCH OF {len(files_todo)} FILE(S)')




"""
-------------------------------------------------------------------------------------------
le programme commence ici !
--------------------------------------------------------------------------------------------
"""
if __name__ == '__main__':
    freeze_support() # enables multiprocessing for py-2-exe
        
    # check for CLI input
    if len(sys.argv)>1:
        serfiles.extend(CLI_handler.handle_CLI(options))

    if 0: #test code for performance test
        read_ini()
        serfiles.extend(UI_handler.inputUI(options))
        cProfile.run('handle_files(serfiles, options)', sort='cumtime')
    else:
        # if no command line arguments, open GUI interface
        if len(serfiles)==0:
            # read initial parameters from config.txt file
            read_ini()
            while True:
                newfiles = UI_handler.inputUI(options) # get files
                if newfiles is None:
                    break # end loop
                serfiles.extend(newfiles)
                if options['selected_mode'] == 'File input mode':
                    handle_files(serfiles, options) # handle files
                elif options['selected_mode'] == 'Folder input mode':
                    handle_folder(options)
                else:
                    raise Exception('invalid selected_mode: ' + options['selected_mode'])
                serfiles.clear() # clear files that have been processed
            write_ini()
        else:
            handle_files(serfiles, options, flag_command_line = True) # use inputs from CLI


