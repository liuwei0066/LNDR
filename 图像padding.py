import numpy as np
import os
from tqdm import tqdm  # 如果你没有导入 tqdm，需要先安装它：pip install tqdm


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """

    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


# 你的图像文件夹路径
image_folder = "/path/to/your/image/folder"

# 获取文件夹中的所有图像文件
image_files = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.endswith(('.png', '.jpg', '.jpeg'))]

# 遍历图像文件，并对每一对图像进行填充
for i in tqdm(range(0, len(image_files), 2)):
    imfile1, imfile2 = image_files[i], image_files[i + 1]

    # 加载图像
    image1 = load_image(imfile1)
    image2 = load_image(imfile2)

    # 创建 InputPadder 实例
    padder = InputPadder(image1.shape, divis_by=32)

    # 对图像进行填充
    image1, image2 = padder.pad(image1, image2)

    # 这里可以添加保存填充后图像的代码，例如使用 OpenCV 或 PIL 库
    # save_image(image1, output_path1)
    # save_image(image2, output_path2)
import os
import cv2
from tqdm import tqdm

# 你的图像文件夹路径
image_folder = "/path/to/your/image/folder"
output_folder = "/path/to/your/output/folder"  # 新增：指定保存填充图像的文件夹

# 获取文件夹中的所有图像文件
image_files = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.endswith(('.png', '.jpg', '.jpeg'))]

# 新增：创建保存填充图像的文件夹
os.makedirs(output_folder, exist_ok=True)

# 遍历图像文件，并对每张图像进行填充和保存
for imfile in tqdm(image_files):
    # 加载图像
    image = load_image(imfile)

    # 创建 InputPadder 实例
    padder = InputPadder(image.shape, divis_by=32)

    # 对图像进行填充
    image = padder.pad(image)[0]

    # 新增：构造输出文件路径
    output_path = os.path.join(output_folder, os.path.basename(imfile))

    # 新增：使用 OpenCV 保存填充后的图像
    cv2.imwrite(output_path, cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR))
