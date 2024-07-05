import cv2
import numpy as np
from matplotlib import pyplot as plt

def remove_black_borders(image_path):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # 创建掩膜，黑色部分设为255（需要填充），其他部分设为0
    mask = np.all(image == [0, 0, 0], axis=-1).astype(np.uint8) * 255

    # 使用图像修复算法填充黑边区域
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    cv2.imwrite(image_path, inpainted_image)
if __name__ == '__main__':
    # 示例使用
    image_path = '00110.png'
    result_image = remove_black_borders(image_path)
    cv2.imwrite('result'+image_path, result_image)
    # 显示结果
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))

    plt.subplot(1, 2, 2)
    plt.title('Image without Black Borders')
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))

    plt.show()
