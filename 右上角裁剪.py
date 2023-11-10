from PIL import Image
import os

def crop_image(input_path, output_path, target_size):
    # 打开图像
    image = Image.open(input_path)

    # 获取图像的宽度和高度
    width, height = image.size

    # 计算裁剪框的位置
    left = width - target_size[0]
    top = 0
    right = width
    bottom = target_size[1]

    # 裁剪图像
    cropped_image = image.crop((left, top, right, bottom))

    # 保存裁剪后的图像
    cropped_image.save(output_path)

# 输入文件夹和输出文件夹路径
input_folder = '/home/indemind/datasets/depth_estimate/madnet_test_data/paker-train/cam1'  # 替换为包含所有输入图像的文件夹路径
output_folder = 'right'  # 替换为保存裁剪后图像的文件夹路径

# 目标大小
target_size = (320, 200)  # 替换为你想要的目标大小

# 获取输入文件夹中的所有文件
file_list = os.listdir(input_folder)

# 遍历文件列表并执行裁剪操作
for file_name in file_list:
    # 构建输入和输出文件的完整路径
    input_path = os.path.join(input_folder, file_name)
    output_path = os.path.join(output_folder, file_name)

    # 调用裁剪函数
    crop_image(input_path, output_path, target_size)
