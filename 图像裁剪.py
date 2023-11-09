import cv2
import os

# 定义裁剪的尺寸
width, height = 320, 200

# 读入文件夹路径
src_folder = 'cam1'  #原始文件路径
dst_folder = 'right'   #保存文件的路径

# 选择裁剪的方式 1就是以原点，如果选择2就是左上角，如果选择3就是右上角
mode = 2  #This 选择

# 设置裁剪的位置
if mode == 1:
    start_point = (0, 0)
elif mode == 2:
    start_point = (0, 0)
elif mode == 3:
    start_point = (0, height - 384)
else:
    raise Exception("无效的裁剪方式")

# 创建目标文件夹
if not os.path.exists(dst_folder):
    os.makedirs(dst_folder)

# 遍历文件夹内的所有图片
for file_name in os.listdir(src_folder):
    file_path = os.path.join(src_folder, file_name)

    # 读入图片
    img = cv2.imread(file_path)

    # 裁剪图片
    cropped_img = img[start_point[1]:start_point[1]+height, start_point[0]:start_point[0]+width]

    # 保存裁剪后的图片
    dst_path = os.path.join(dst_folder, file_name)
    cv2.imwrite(dst_path, cropped_img)

