import os
import pandas as pd
import numpy as np
import time
import cv2
import scipy.io
def mix_prepare():
    Expression_version = 'Mix_DFME_extend'
    Version = '22'

    DATA_FILE = '../{}_data_{}.txt'.format(Expression_version, Version)
    LABEL_FILE = '../{}_label_{}.txt'.format(Expression_version, Version)
    SUBJECT_FILE = '../{}_subject_{}.txt'.format(Expression_version, Version)
    FLOW_FILE = '../{}_flow_{}.txt'.format(Expression_version, Version)
    AFFLOW_FILE = '../{}_afflow_{}.txt'.format(Expression_version, Version)
    label_files=open(LABEL_FILE,mode='w')
    subject_files=open(SUBJECT_FILE,mode='w')
    afflow_files=open(AFFLOW_FILE,mode='w')

# # =========================================================================================
    apex_path = './SAMM_Micro.xlsx'
    SAMM_path = './SAMM/'
    save_path = './SAMM_landmarks_2pic/'
    flow_path = './SAMM_motion_flow/'
    # 读取xlsx文件
    df = pd.read_excel(apex_path, header=0)
    emotion_dict = {
        "Anger": 0,
        "Contempt": 1,
        "Disgust": 2,
        "Fear": 3,
        "Happiness": 4,
        "Sadness": 5,
        "Surprise": 6
    }
    # 遍历 DataFrame 中的每一行
    for index, row in df.iterrows():
        sub = row['Subject']  # 假设 Excel 中有一列叫做 'sub'
        seq = row['Filename']  # 假设 Excel 中有一列叫做 'seq'
        emotion = row['Estimated Emotion']  # 假设 Excel 中有一列叫做 'emotion'

        # 构建文件路径
        file_path = f"../SAMM_motion_flow/{str(sub).zfill(3)}/{seq}/Merge.jpg"
        print(file_path)
        # 检查文件是否存在
        if os.path.exists(file_path) and emotion in emotion_dict:
            # 写入 Mix_casme3_macro_afflow.txt
            relative_path = f"../SAMM_motion_flow/{str(sub).zfill(3)}/{seq}/Merge.jpg"
            afflow_files.write(relative_path + '\n')

            # 将情绪转换为数字并写入 Mix_casme3_macro_label.txt
            emotion_label = emotion_dict.get(emotion, -1)
            label_files.write(str(emotion_label) + '\n')

            # 写入 'Macro' 到 Mix_casme3_macro_subject.txt
            subject_files.write('Micro\n')

   
# =================================================================================================
    apex_path = './CAS(ME)2.xlsx'
    # 读取xlsx文件
    df = pd.read_excel(apex_path, header=0)    
    emotion_dict = {
        "anger": 0,
        "contempt": 1,
        "disgust": 2,
        "fear": 3,
        "happiness": 4,
        "sadness": 5,
        "surprise": 6
    }
    # 遍历 DataFrame 中的每一行
    for index, row in df.iterrows():
        sub = row['sub']  # 假设 Excel 中有一列叫做 'sub'
        seq = row['seq']  # 假设 Excel 中有一列叫做 'seq'
        emotion = row['emotion']  # 假设 Excel 中有一列叫做 'emotion'
        types = row['type']  # 假设 Excel 中有一列叫做 'type'
        if types == 'macro-expression':
            # 构建文件路径
            file_path = f"../ME2-Macro_flow/{sub}/{seq}/Merge.jpg"
            print(file_path)
            # 检查文件是否存在
            if os.path.exists(file_path) and emotion in emotion_dict:
                # 写入 Mix_casme3_macro_afflow.txt
                relative_path = f"../ME2-Macro_flow/{sub}/{seq}/Merge.jpg"
                afflow_files.write(relative_path + '\n')

                # 将情绪转换为数字并写入 Mix_casme3_macro_label.txt
                emotion_label = emotion_dict.get(emotion, -1)
                label_files.write(str(emotion_label) + '\n')

                # 写入 'Macro' 到 Mix_casme3_macro_subject.txt
                subject_files.write('Macro\n')
        else:
            # 构建文件路径
            file_path = f"../ME2_flow/{sub}/{seq}/Merge.jpg"

            # 检查文件是否存在
            if os.path.exists(file_path) and emotion in emotion_dict:
                # 写入 Mix_casme3_macro_afflow.txt
                relative_path = f"../ME2_flow/{sub}/{seq}/Merge.jpg"
                afflow_files.write(relative_path + '\n')

                # 将情绪转换为数字并写入 Mix_casme3_macro_label.txt
                emotion_label = emotion_dict.get(emotion, -1)
                label_files.write(str(emotion_label) + '\n')

                # 写入 'Macro' 到 Mix_casme3_macro_subject.txt
                subject_files.write('Micro\n')
    


# =======================================================================================================

    label_path = '../CK+/Emotion/'
    flow_path = '../CK_flow/'

    for subject in os.listdir(label_path):
        subject_label_path = os.path.join(label_path, subject)
        for test in os.listdir(subject_label_path):
            test_label_path = os.path.join(subject_label_path, test)
            if os.listdir(test_label_path):

                with open (os.path.join(test_label_path, os.listdir(test_label_path)[-1]), 'r') as rl:
                    label = rl.readline()
                    label = int(float(label.strip('\n')))

                if label != 0:
#                    with open (LABEL_FILE, 'a') as l:
#                        l.write(str(CK_dict[label]) + '\n')
                    file_path = os.path.join(flow_path, subject, test,'Merge.jpg')
                    print(file_path)
                    if os.path.exists(file_path):
                        label_files.writelines(str(label-1) + '\n')
                        subject_files.writelines('Macro\n')
                        afflow_files.writelines(file_path + '\n')
    
 
    label_files.close()
    subject_files.close()
    afflow_files.close()



if __name__ == '__main__':
    start = time.time()
    mix_prepare()
    print('cost: ', time.time() - start)
