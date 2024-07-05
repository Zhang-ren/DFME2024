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
EXPRESSION_VERSION = 'Mix_DFME_MMEW_macro'
VERSION = '22'
DATA_FILE = f'../{EXPRESSION_VERSION}_data_{VERSION}.txt'
LABEL_FILE = f'../{EXPRESSION_VERSION}_label_{VERSION}.txt'
SUBJECT_FILE = f'../{EXPRESSION_VERSION}_subject_{VERSION}.txt'
FLOW_FILE = f'../{EXPRESSION_VERSION}_flow_{VERSION}.txt'
AFFLOW_FILE = f'../{EXPRESSION_VERSION}_afflow_{VERSION}.txt'

DFME_PATH = '../MMEW/Macro_Expression'
DFME_SAVE_PATH = '../MMEW_macro_mag_landmarks/'
DFME_FLOW_PATH = '../MMEW_macro_mag_motion_flow/'

def mix_prepare():
    emotion_dict = {
        "anger": 0,
        "contempt": 1,
        "disgust": 2,
        "fear": 3,
        "happiness": 4,
        "sadness": 5,
        "surprise": 6
    }
    
    data_lines, label_lines, subject_lines, flow_lines, afflow_lines = [], [], [], [], []
    for root, dirs, files in os.walk(DFME_PATH):
        for dir in dirs:
            dir_paths = os.path.join(root, dir)
            for dir_path in os.listdir(dir_paths):
                save_test_path = os.path.join(DFME_SAVE_PATH, dir, dir_path.split('_')[0])
                
                flow_test_path = os.path.join(DFME_FLOW_PATH, dir, dir_path.split('_')[0])

                dir_path = os.path.join(dir_paths, dir_path)
                if not os.path.isdir(dir_path):
                    continue
                image_files = [f for f in os.listdir(dir_path) if f.endswith(('.jpg', '.png'))]
                image_files.sort()  # 确保文件是按照名称排序的key=lambda x:int(x.split('.')[0])
                print(image_files)
                if image_files:
                    onset_pic = image_files[0]
                    apex_pic = image_files[4]
                    onset_path = os.path.join(dir_path, onset_pic)
                    apex_path = os.path.join(dir_path, apex_pic)
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
                    print(dir_path)

                    data_lines.extend([onset_save_path + '\n', apex_save_path + '\n'])                    
                    label_lines.append(str(emotion_dict[os.path.basename(dir_path)]) + '\n')
                    subject_lines.append('Micro' + '\n')
                    flow_lines.append(Flow_pic_path + '\n')
                    afflow_lines.append(AFFlow_pic_path + '\n')

                write_to_file(DATA_FILE, data_lines)
                write_to_file(LABEL_FILE, label_lines)
                write_to_file(SUBJECT_FILE, subject_lines)
                write_to_file(FLOW_FILE, flow_lines)
                write_to_file(AFFLOW_FILE, afflow_lines)
            
    
def get_landmarks(image, det, predictor):
    x_coords = np.zeros((len(det), 68), dtype=int)
    y_coords = np.zeros((len(det), 68), dtype=int)
    for i, d in enumerate(det):
        landmarks = predictor(image, d)
        for idx, point in enumerate(landmarks.parts()):
            x_coords[i][idx] = point.x
            y_coords[i][idx] = point.y
    return x_coords, y_coords
def crop_pic(onset_path, apex_path, onset_save_path, apex_save_path, flow_save_path, afflow_pic_path):
    size = 320
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    onset_image = cv2.resize(cv2.imread(onset_path), (size, size))
    apex_image = cv2.resize(cv2.imread(apex_path), (size, size))
    onset_det = detector(onset_image, 0)
       
    

    
    x1, y1 = get_landmarks(onset_image, onset_det, predictor)
    if not len(x1) or not len(y1):
        print('No face detected')
        log_error(onset_path, apex_path)
        return False
    # 将 onset_path图片拷贝到 onset_save_path
    # 将 apex_path图片拷贝到 apex_save_path
    cv2.imwrite(onset_save_path, onset_image)
    cv2.imwrite(apex_save_path, apex_image)
    x, y = adjust_coordinates(x1, y1, size)
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
