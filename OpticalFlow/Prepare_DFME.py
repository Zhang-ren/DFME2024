import os
import pandas as pd
import numpy as np
import dlib
import time
from PIL import Image
import cv2
from dlib68 import getxy
import scipy.io
from limited import limits
from apexs import crop_face, imflow
from fill import remove_black_borders
mode = 'test'
if mode == 'train':
    # Constants
    EXPRESSION_VERSION = 'Mix_DFME_10'
else:
    EXPRESSION_VERSION = 'Mix_DFME_test_10B'
VERSION = '22'
DATA_FILE = f'../{EXPRESSION_VERSION}_data_{VERSION}.txt'
LABEL_FILE = f'../{EXPRESSION_VERSION}_label_{VERSION}.txt'
SUBJECT_FILE = f'../{EXPRESSION_VERSION}_subject_{VERSION}.txt'
FLOW_FILE = f'../{EXPRESSION_VERSION}_flow_{VERSION}.txt'
AFFLOW_FILE = f'../{EXPRESSION_VERSION}_afflow_{VERSION}.txt'
if mode == 'train':
    DFME_PATH = '../DFME2024_5'
    DFME_SAVE_PATH = '../DFME_mag_landmarks_10/'
    DFME_FLOW_PATH = '../DFME_mag_motion_flow_10/'
    EXCEL_FILE = './CCAC2024.xlsx'
else:
    DFME_PATH = '../DFME_B_10' #'/home/data2/MEGC/DFME2024_test_5/'
    DFME_SAVE_PATH = '../DFME_mag_test_landmarks_10B/'
    DFME_FLOW_PATH = '../DFME_mag_test_motion_flow_10B/'
    EXCEL_FILE = './CCAC2024_B.xlsx'

def mix_prepare():
    data = pd.read_excel(EXCEL_FILE, header=0)

    data_lines, label_lines, subject_lines, flow_lines, afflow_lines = [], [], [], [], []

    for _, row in data.iterrows():
        subject = row['Filename']
        # test_path = os.path.join(DFME_PATH, subject)
        test_path = os.path.join(DFME_PATH, subject + '_fl0.04_fh0.4_fs30.0_n2_differenceOfIIR')
        save_test_path = os.path.join(DFME_SAVE_PATH, subject)
        flow_test_path = os.path.join(DFME_FLOW_PATH, subject)

        pic_list = sorted(os.listdir(test_path))
        onset_pic = pic_list[0]
        apex_pic = pic_list[len(pic_list) // 2]
        onset_path = os.path.join(test_path, onset_pic)
        apex_path = os.path.join(test_path, apex_pic)
        onset_save_path = os.path.join(save_test_path, onset_pic)
        apex_save_path = os.path.join(save_test_path, apex_pic)

        if not os.path.exists(flow_test_path):
            os.makedirs(flow_test_path)

        Flow_pic_path = os.path.join(flow_test_path, 'motion_flow.jpg')
        AFFlow_pic_path = os.path.join(flow_test_path, 'Merge.jpg')

        if not os.path.exists(save_test_path):
            os.makedirs(save_test_path)

        if not os.path.exists(AFFlow_pic_path):
            flag = crop_pic(onset_path, apex_path, onset_save_path, apex_save_path, Flow_pic_path, AFFlow_pic_path)
            if not flag:
                continue
        print(test_path)

        data_lines.extend([onset_save_path + '\n', apex_save_path + '\n'])
        if mode == 'train':
            label_lines.append(str(row['EmotionId']) + '\n')
        else:
            label_lines.append(str(0) + '\n')
        subject_lines.append(row['Subject'] + '\n')
        flow_lines.append(Flow_pic_path + '\n')
        afflow_lines.append(AFFlow_pic_path + '\n')

        write_to_file(DATA_FILE, data_lines)
        write_to_file(LABEL_FILE, label_lines)
        write_to_file(SUBJECT_FILE, subject_lines)
        write_to_file(FLOW_FILE, flow_lines)
        write_to_file(AFFLOW_FILE, afflow_lines)

def crop_pic(onset_path, apex_path, onset_save_path, apex_save_path, flow_save_path, afflow_pic_path):
    sizea = 340
    onset_crop, apex_crop, sizea, onx, ony, apx, apy = crop_face(onset_path, apex_path, sizea)
    if onset_crop is None or apex_crop is None:
        print('No face detected')
        log_error(onset_path, apex_path)
        return False
    
    size = 320
    onset_crop = crop_resize(onset_crop, onx, ony, size, sizea)
    apex_crop = crop_resize(apex_crop, apx, apy, size, sizea)

    onset_cropped = Image.fromarray(cv2.cvtColor(onset_crop, cv2.COLOR_BGR2RGB))
    apex_cropped = Image.fromarray(cv2.cvtColor(apex_crop, cv2.COLOR_BGR2RGB))
    onset_cropped.save(onset_save_path)
    apex_cropped.save(apex_save_path)
    remove_black_borders(onset_save_path)
    remove_black_borders(apex_save_path)
    x, y = adjust_coordinates(onx, ony, size)
    imflow(x, y, onset_save_path, apex_save_path, flow_save_path, afflow_pic_path)
    return True

def crop_resize(image, x_points, y_points, size, sizea):
    xl, xr, yl, yr = limits(x_points, y_points, size, sizea)
    cropped_image = image[yl:yr, xl:xr]
    return cv2.resize(cropped_image, (size, size))

def adjust_coordinates(x_points, y_points, size):
    x = np.clip(x_points, 1, size-1)
    y = np.clip(y_points, 1, size-1)
    return np.array(x), np.array(y)

def log_error(onset_path, apex_path):
    with open(EXPRESSION_VERSION +'_error_log.txt', 'a') as f:
        f.write(onset_path + '\n')
        f.write(apex_path + '\n')

def write_to_file(file_path, lines):
    with open(file_path, 'w') as file:
        file.writelines(lines)

if __name__ == '__main__':
    start = time.time()
    mix_prepare()
    print('Cost: ', time.time() - start)
