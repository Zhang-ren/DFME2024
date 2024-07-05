import os
import dlib
import cv2
import transplant
import numpy as np
import operator
from dlib68 import getxy

matlab = transplant.Matlab(jvm=False, desktop=False)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def get_landmarks(image, det):
    x_coords = np.zeros((len(det), 68), dtype=int)
    y_coords = np.zeros((len(det), 68), dtype=int)
    for i, d in enumerate(det):
        landmarks = predictor(image, d)
        for idx, point in enumerate(landmarks.parts()):
            x_coords[i][idx] = point.x
            y_coords[i][idx] = point.y
    return x_coords, y_coords

def find_largest_face(det, x_coords):
    max_width = 0
    index = 0
    for i, d in enumerate(det):
        width = max(x_coords[i]) - min(x_coords[i])
        if width > max_width:
            max_width = width
            index = i
    return index

def transform_landmarks(landmarks, transform):
    transformed = np.zeros_like(landmarks)
    for i in range(landmarks.shape[1]):
        x, y = landmarks[0, i], landmarks[1, i]
        point = np.array([x, y, 1])
        transformed_point = np.dot(transform, point)
        transformed[0, i] = transformed_point[0]
        transformed[1, i] = transformed_point[1]
    return transformed

def crop_face(onset_path, apex_path, size=340):
    onset_image = cv2.imread(onset_path)
    apex_image = cv2.imread(apex_path)

    onset_det = detector(onset_image, 0)
    apex_det = detector(apex_image, 0)
    if len(onset_det) == 0 or len(apex_det) == 0:
        return None, None, None, None, None, None, None
    x1, y1 = get_landmarks(onset_image, onset_det)
    x2, y2 = get_landmarks(apex_image, apex_det)

    inx = find_largest_face(onset_det, x1)
    ina = find_largest_face(apex_det, x2)

    onset_faces = dlib.full_object_detections()
    apex_faces = dlib.full_object_detections()

    onset_faces.append(predictor(onset_image, onset_det[inx]))
    apex_faces.append(predictor(apex_image, apex_det[ina]))

     # 使用 dlib.get_face_chip 单独裁剪每个面部
    onset_crop = dlib.get_face_chip(onset_image, onset_faces[0], size=size, padding=0.25)
    apex_crop = dlib.get_face_chip(apex_image, apex_faces[0], size=size, padding=0.25)

    # 手动计算仿射变换矩阵
    def estimate_affine_transform(src_landmarks, dst_landmarks):
        src_points = np.array([src_landmarks[0], src_landmarks[1]]).T
        dst_points = np.array([dst_landmarks[0], dst_landmarks[1]]).T
        transform, _ = cv2.estimateAffinePartial2D(src_points, dst_points)
        return np.vstack([transform, [0, 0, 1]])

    # 估算初始图像和裁剪图像之间的仿射变换
    initial_landmarks = np.vstack((x1[inx], y1[inx], np.ones(68)))
    cropped_landmarks = np.vstack((get_landmarks(onset_crop, [dlib.rectangle(0, 0, onset_crop.shape[1], onset_crop.shape[0])])[0][0], get_landmarks(onset_crop, [dlib.rectangle(0, 0, onset_crop.shape[1], onset_crop.shape[0])])[1][0], np.ones(68)))
    onset_transform = estimate_affine_transform(initial_landmarks, cropped_landmarks)

    initial_landmarks = np.vstack((x2[ina], y2[ina], np.ones(68)))
    cropped_landmarks = np.vstack((get_landmarks(apex_crop, [dlib.rectangle(0, 0, apex_crop.shape[1], apex_crop.shape[0])])[0][0], get_landmarks(apex_crop, [dlib.rectangle(0, 0, apex_crop.shape[1], apex_crop.shape[0])])[1][0], np.ones(68)))
    apex_transform = estimate_affine_transform(initial_landmarks, cropped_landmarks)

    # 变换初始关键点到裁剪后的图像
    transformed_onset_landmarks = transform_landmarks(initial_landmarks, onset_transform)
    transformed_apex_landmarks = transform_landmarks(initial_landmarks, apex_transform)

    onx, ony = transformed_onset_landmarks[0], transformed_onset_landmarks[1]
    apx, apy = transformed_apex_landmarks[0], transformed_apex_landmarks[1]
    if len(onx) == 0 or len(ony) == 0 or len(apx) == 0 or len(apy) == 0:
        return None, None, None, None, None, None, None

    return onset_crop, apex_crop, size, onx.astype(int), ony.astype(int), apx.astype(int), apy.astype(int)
def apexs(test_image_path, image_dir, mode=0, find=0):
    rads = []
    onset_path = os.path.join(test_image_path, image_dir[0])
    if mode == 0:
        for apex_image in image_dir[1:]:
            apex_path = os.path.join(test_image_path, apex_image)
            onset_crop, apex_crop, size, _, _, _, _ = crop_face(onset_path, apex_path)
            ox, oy = getxy(onset_crop, size)
            rad = matlab.maxflow(onset_crop, apex_crop, ox, oy, 0)
            rads.append(rad)
        max_index = np.argmax(rads)
        return max_index
    else:
        low, high = 1, len(image_dir) - 1
        used_apex = {}
        while low <= high:
            apex_num = (low + high) // 2
            apex_path = os.path.join(test_image_path, image_dir[apex_num])
            onset_crop, apex_crop, size, _, _, _, _ = crop_face(onset_path, apex_path, 320)
            ox, oy = getxy(onset_crop, size)
            rad = matlab.maxflow(onset_crop, apex_crop, ox, oy, 0)
            if 8 <= rad <= 11:
                print(f'{rad} find')
                return apex_num
            elif rad < 8:
                low = apex_num + 1
            else:
                high = apex_num - 1
            used_apex[apex_num] = abs(rad - 9.5)

        if find == 0:
            print(rad)
            return apex_num
        else:
            min_key = min(used_apex, key=used_apex.get)
            print(used_apex[min_key] + 9.5)
            return min_key

def imflow(x, y, onset_save_path, apex_save_path, flow_save_path, afflow_pic_path):
    return matlab.test(x, y, onset_save_path, apex_save_path, flow_save_path, afflow_pic_path)

def maxrad(onset_save_path, apex_save_path):
    onset_crop = cv2.imread(onset_save_path)
    apex_crop = cv2.imread(apex_save_path)
    ox, oy = getxy(onset_crop, 320)
    rad = matlab.maxflow(onset_crop, apex_crop, ox, oy, 0)
    return rad

if __name__ == '__main__':
    test_image_path = '/home/halo/音乐/MEGC2019-TIMED/CK+/cohn-kanade-images/S005/001'
    image_dir = [f for f in sorted(os.listdir(test_image_path)) if f.endswith('.png')]
    s = apexs(test_image_path, image_dir, 1, 1)
    print(s)
