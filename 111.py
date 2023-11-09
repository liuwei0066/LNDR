def validate_kitti(model, iters=32, mixed_prec=False):
    """ 使用KITTI-2015（训练集）拆分进行验证 """
    model.eval()  # 设置模型为评估模式，不进行训练
    aug_params = {}  # 数据增强参数（在此处未使用）
    val_dataset = datasets.KITTI(aug_params, image_set='training')  # 创建KITT数据集对象，用于验证
    torch.backends.cudnn.benchmark = True  # 使用CUDNN库进行性能优化

    out_list, epe_list, elapsed_list = [], [], []  # 创建空列表，用于存储结果
    for val_id in range(len(val_dataset)):  # 遍历验证集中的样本
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]  # 从数据集中获取图像和流场标签

        # 将图像转移到GPU上并添加一个维度
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        # 图像填充，确保图像尺寸能够被32整除

        with autocast(enabled=mixed_prec):  # 使用混合精度训练（在此处未使用）
            start = time.time()
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)  # 使用模型进行流场估计
            end = time.time()

        if val_id > 50:
            elapsed_list.append(end - start)  # 记录计算时间

        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)  # 移除填充并将结果从GPU移到CPU

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)  # 断言确保估计的流场尺寸与标签一致
        epe = torch.sum((flow_pr - flow_gt) ** 2, dim=0).sqrt()  # 计算端点误差（EPE）

        epe_flattened = epe.flatten()
        val = valid_gt.flatten() >= 0.5  # 获取有效像素标志

        out = (epe_flattened > 3.0)  # 判断EPE是否大于3.0
        image_out = out[val].float().mean().item()  # 计算图像上的D1错误率
        image_epe = epe_flattened[val].mean().item()  # 计算图像上的平均EPE

        if val_id < 9 or (val_id + 1) % 10 == 0:
            logging.info(
                f"KITTI Iter {val_id + 1} out of {len(val_dataset)}. EPE {round(image_epe, 4)} D1 {round(image_out, 4)}. Runtime: {format(end - start, '.3f')}s ({format(1 / (end - start), '.2f')}-FPS)")  # 打印验证进度和结果

        epe_list.append(epe_flattened[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)  # 计算平均EPE
    d1 = 100 * np.mean(out_list)  # 计算D1错误率

    avg_runtime = np.mean(elapsed_list)  # 计算平均运行时间

    print(f"Validation KITTI: EPE {epe}, D1 {d1}, {format(1 / avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")  # 打印验证结果

    return {'kitti-epe': epe, 'kitti-d1': d1}  # 返回验证结果作为字典
