from PIL import Image
import os

def center_crop_and_save_images(input_folder, output_folder, target_size):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹中所有图像文件
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    for image_file in image_files:
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)

        # 打开图像
        image = Image.open(input_path)

        # 获取图像的中心坐标
        width, height = image.size
        left = (width - target_size[0]) // 2
        top = (height - target_size[1]) // 2
        right = (width + target_size[0]) // 2
        bottom = (height + target_size[1]) // 2

        # 中心裁剪图像
        cropped_image = image.crop((left, top, right, bottom))

        # 保存裁剪后的图像
        cropped_image.save(output_path)

        print(f"Image {image_file} center-cropped and saved to {output_path}")

# 指定输入文件夹、输出文件夹和目标尺寸
input_folder = "paker320_train10/cv"
output_folder = "paker320_10"
target_size = (320, 200)  # 替换为你想要的尺寸

# 调用函数进行中心裁剪和保存
center_crop_and_save_images(input_folder, output_folder, target_size)
