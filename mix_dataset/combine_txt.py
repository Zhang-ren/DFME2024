import os

# 定义要读取的文件名列表
file_names = ['Mix_DFME_10_afflow_22.txt', 'Mix_casme3_macro_afflow.txt', 'Mix_casme3_micro_afflow.txt', 'Mix_DFME_extend_afflow_22.txt','Mix_DFME_10_gen_afflow_22.txt', 'Mix_DFME_MMEW_afflow_22.txt', 'Mix_DFME_MMEW_macro_afflow_22.txt']
types = ['afflow','label','subject']
# 定义输出文件名
output_file_name = '../combined_'

# 打开输出文件
for ty in types:
    with open(output_file_name+ty+'.txt', 'w', encoding='utf-8') as output_file:
        # 遍历每个输入文件
        for file_name in file_names:
            # 打开并读取当前文件
            with open(file_name.replace('afflow',ty) , 'r', encoding='utf-8') as input_file:
                # 读取文件内容并写入输出文件
                content = input_file.read()
                output_file.write(content)

print(f"All files have been combined into {output_file_name}")
