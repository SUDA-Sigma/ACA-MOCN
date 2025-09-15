import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from models.counters.HKINet import HKINet
import os
from models.counters.MCC import MCC
from models.counters.MCNN import MCNN
from models.counters.MFANet import MFANet
import torch.nn.functional as F
import torchvision.transforms as transforms
from misc.layer import Gaussianlayer
# from matplotlib.font_manager import FontProperties


# import Image

# device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # 加载原图
# def load_original_image(image_path):
#     return cv2.imread(image_path)

# # 加载模型
# # def load_model(model_path):
# #     # 假设你的模型定义为MyModel
# #     model = torch.load(model_path)
# #     model.eval()
# #     return model

# # 加载模型
# def load_model(model_path):
#     # 假设你的模型定义为MyModel
#     model = HKINet().to(device)
#     # model = MCC().to(device)
#     ckpt  = torch.load(model_path, map_location=device)
#     state = ckpt.get('state_dict', ckpt)
#     state = {k.replace('module.', ''): v for k,v in state.items()}
#     state = {k.replace('CCN.', ''): v for k, v in state.items()}
#     # model.load_state_dict(state)
#     model.load_state_dict(state, strict=False)
#     model.eval()

#     # model = HKINet()
#     # model = torch.load(model_path)
#     # model.eval()
#     return model


# # 生成热力图
# def generate_heatmap(array):
#     # heatmap = torch.sum(array)    #所有通道求和
#     # max_value = torch.max(heatmap)
#     # min_value = torch.min(heatmap)
#     # heatmap = (heatmap-min_value)/(max_value-min_value)*255
    
#     # heatmap = heatmap.cpu().numpy().astype(np.uint8).transpose(1,2,0)  # 提取热力图
    
#     # # 将矩阵转换为image类
#     # heatmap=cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)


#     # 将数组归一化到0-1范围
#     normalized = (array - array.min()) / (array.max() - array.min())
#     # 转换为热力图
#     print("normalized.sum()",normalized.sum())

#     heatmap = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
#     # heatmap = cv2.cvtColor(heatmap)
#     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#     return heatmap

# # 主函数
# def main():
#     # 文件路径（替换为你的实际路径）
#     # original_image_path = '/home/zzw/data/NWPU-MOC/images/A0_2020_orth25_7_6_3.jpg'
#     # annotation_path = '/home/zzw/data/NWPU-MOC/gt/A0_2020_orth25_7_6_3.npz'
#     # model_path = '/home/zzw/data/NWPU-MOC/weights.pth'  # 替换为你的模型权重路径
#     # output_dir = '/home/zzw/data/NWPU-MOC/output/'
#     # A2_2020_orth25_27_31_3.png
#     # A0_2020_orth25_7_6_3.png
#     original_image_path = '/home/zzw/data/NWPU-MOC/rgb/A2_2020_orth25_27_31_3.png'
#     annotation_path = '/home/zzw/data/NWPU-MOC/gt/A2_2020_orth25_27_31_3.npz'
#     model_path = '/home/zzw/code/HKINet-NWPU/exp/04-06_23-59_MOC_RS_HKINet_loss_MRCloss_alpha_1_base_lr_5e-05_NIR=False/all_ep_185_cls_avg_mae_5.6_cls_avg_mse_11.3_cls_weight_mse_52.3.pth'
#     # model_path = '/home/zzw/data/pretrain/FFPN_ACA_NWPU-MOC_best.pth'
#     output_dir = '/home/zzw/code/HKINet-NWPU/'


#     # 加载原图
#     # original_image = load_original_image(original_image_path)

#     # 加载标注文件
#     annotations = np.load(annotation_path)['arr_0']  # 假设键为'arr_0'

#     # 加载模型并进行推理
#     model = load_model(model_path)

#     transform = transforms.Compose([transforms.ToTensor(),
#                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
#     image = cv2.imread(original_image_path)     #image_path为文件路径
#     input_tensor = transform(image)    #将图片转换为tensor类型
#     input_tensor = input_tensor.unsqueeze(0) 

#     # input_tensor = torch.from_numpy(original_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
#     input_tensor = input_tensor.to('cuda')  # 如果模型在GPU上，确保输入也在GPU上
#     with torch.no_grad():
#         output = model(input_tensor)  # 模型输出形状假设为[1, 6, 1024, 1024]
#     output_maps = output.squeeze(0).cpu().numpy()  # 移除批次维度，并移动到CPU

#     # 创建一个两行七列的子图布局
#     fig, axs = plt.subplots(2, 7, figsize=(20, 5))

#     # 第一行：原图的各个通道热力图
#     for i in range(6):
#         ax = axs[0, i+1]  # 第一行从第二列开始
#         if i < annotations.shape[2]:
#             annotation_map = annotations[:, :, i]
#             print("annotation_map.sum()", annotation_map.sum())
#             heatmap = generate_heatmap(annotation_map)
#             ax.imshow(heatmap)
#             ax.set_title(f'Annotation Channel {i+1}')
#             ax.axis('off')
#         else:
#             ax.axis('off')

#     # 第二行：模型输出的各个通道热力图
#     for i in range(6):
#         ax = axs[1, i+1]  # 第二行从第二列开始
#         if i < output_maps.shape[0]:
#             output_map = output_maps[i, :, :]
#             print(output_map.sum())
#             heatmap = generate_heatmap(output_map)
#             sum_value = np.sum(output_map)
#             print("sum_value", sum_value)
#             # 在热力图上标注数值
#             ax.imshow(heatmap)
#             ax.text(10, 20, f'Sum: {sum_value:.2f}', color='white', fontsize=8)
#             ax.set_title(f'Model Output Channel {i+1}')
#             ax.axis('off')
#         else:
#             ax.axis('off')

#     # 第一列：原图
#     axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     axs[0, 0].set_title('Original Image')
#     axs[0, 0].axis('off')

#     # 调整子图间距
#     plt.tight_layout()

#     # 保存最终结果
#     plt.savefig(f'{output_dir}/combined_result_2.png', dpi=100)
#     plt.close()

# if __name__ == "__main__":
#     main()




def load_model(model_path, device):
    
    # model = MCNN().to(device)
    model = MCC().to(device)
    # model = MFANet().to(device)
    # model = HKINet().to(device)
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt.get('state_dict', ckpt)
    # 如果有前缀需要去除，可在此处做 name.replace
    state = {k.replace('module.', ''): v for k, v in state.items()}
    state = {k.replace('CCN.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    # model.load_state_dict(state)
    model.eval()
    return model

def generate_density_maps(model, image_path, device):
    # 载入原图并预处理
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    inp = transform(img_rgb).unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        out = model(inp)                  # [1, C, H, W]
    print(out[:,0,:,:].sum())
    print(out[:,1,:,:].sum())
    print(out[:,2,:,:].sum())
    print(out[:,3,:,:].sum())
    print(out[:,4,:,:].sum())
    print(out[:,5,:,:].sum())
    channel_sums = [float(out[:,i,:,:].sum()) for i in range(6)]
  
    return img_rgb, out.squeeze(0).cpu().numpy(), channel_sums

def multi_class_gauss_map_generate(gt_map):
        class_num = gt_map.shape[1]
        class_gauss_maps = []
        for i in range(class_num):
            class_gt_map = torch.unsqueeze(gt_map[:,i,:,:], 1)
            class_gauss_map = torch.squeeze(Gaussianlayer(class_gt_map), 1)
            class_gauss_maps.append(class_gauss_map)
        gauss_map = torch.stack(class_gauss_maps,1)
        return gauss_map


def smooth_gt_opencv(gt_maps, ksize=15, sigma=4):
    """
    对每个通道应用 GaussianBlur。
    ksize: 内核大小，必须是奇数，如 3,5,7…
    sigma: 标准差，控制平滑程度
    """
    C, H, W = gt_maps.shape
    smoothed = np.zeros_like(gt_maps, dtype=np.float32)
    for i in range(C):
        # GaussianBlur 要求输入为 H×W 单通道灰度图
        single = gt_maps[i].astype(np.float32)
        blurred = cv2.GaussianBlur(single, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        smoothed[i] = blurred
    return smoothed


def plot_with_colorbars(img, gt_maps, pred_maps, save_path, name, channel_sums):
    """
    img:       (H,W,3) 原始 RGB 图
    gt_maps:   (C, H, W) 真实密度图
    pred_maps: (C, H, W) 预测密度图
    """
    C, H, W = gt_maps.shape
    # 统一各通道的色标范围
    # pred_maps = pred_maps / 255 * 3;

    # 每个通道逐一归一化
    # pred_norm = pred_maps
    # for i in range(6):
    #     channel = pred_maps[i]
    #     ch_min, ch_max = channel.min(), channel.max()
    #     channel_norm = (channel - ch_min) / (ch_max - ch_min + 1e-6)
    #     pred_norm[i] = channel_norm
    
    gt_norm = gt_maps
    for i in range(6):
        channel = gt_maps[i]
        ch_min, ch_max = channel.min(), channel.max()
        channel_norm = (channel - ch_min) / (ch_max - ch_min + 1e-6)
        gt_norm[i] = channel_norm


    p_min = pred_maps.min()
    p_max = pred_maps.max()
    pred_norm = (pred_maps - p_min) / (p_max - p_min) 
    
    # g_min = gt_maps.min()
    # g_max = gt_maps.max()
    # gt_norm = (gt_maps - g_min) / (g_max - g_min) 
    # vmin = 0
    # vmax = 1

    # vmin = min(gt_maps.min(), pred_maps.min())
    # vmax = max(gt_maps.max(), pred_maps.max())
    vmin = min(gt_norm.min(), pred_norm.min())
    vmax = max(gt_norm.max(), pred_norm.max())
  
    
    fig, axes = plt.subplots(2, C+1, figsize=((C+1)*3, 2*3))
    # 关闭所有坐标轴
    for ax in axes.flatten():
        ax.axis('off')
    
    # 第一行，第一列：原图
    axes[0,0].imshow(img)
    axes[0,0].axis('off')
    axes[0,0].set_title('Original')

    real_sums = [0, 101, 31, 0, 86, 0]
    # real_sums = [0, 2010, 1, 0, 0, 0]
    
    # 绘制真实密度图
    for i in range(6):
        ax = axes[0, i+1]
        im = ax.imshow(gt_norm[i], cmap='jet', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(f'GT Class {i+1}')
        sum_text = f'{real_sums[i]}'
        ax.text(0.9, 0.1, sum_text,
            transform=ax.transAxes,      # 以 Axes 坐标 (0~1) 为基准
            ha='right', va='bottom',     # 右对齐，底部对齐
            color='white',               # 白色字体
            fontsize=16,
            fontweight='bold',   
            # bbox=dict(facecolor='black', alpha=0.3, pad=1, edgecolor='none')
           )
        # colorbar
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=6)
    
    # 第二行留空第一列
    axes[1,0].imshow(img, aspect='auto')
    axes[1,0].axis('off')
    
    # 绘制预测密度图
    for i in range(6):
        ax = axes[1, i+1]
        im = ax.imshow(pred_norm[i], cmap='jet', vmin=vmin, vmax=vmax, aspect='auto')
        print("pred_norm[i]", pred_norm[i].sum())
        # ax.set_title(f'Pred Class {i+1}')
        sum_text = f'{channel_sums[i] / 6 :.1f}'
        ax.text(0.9, 0.1, sum_text,
            transform=ax.transAxes,      # 以 Axes 坐标 (0~1) 为基准
            ha='right', va='bottom',     # 右对齐，底部对齐
            color='white',               # 白色字体
            fontsize=16,
            fontweight='bold',  
            # bbox=dict(facecolor='black', alpha=0.3, pad=1, edgecolor='none')
           )
    
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=6)
        
    
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{name}_density_comparison.png'), dpi=300)
    plt.close(fig)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # image_path  = '/home/zzw/data/NWPU-MOC/rgb/A2_2020_orth25_27_31_3.png'
    image_path  = '/home/zzw/data/NWPU-MOC/rgb/A0_2020_orth25_7_6_3.png'
    # annotation_path = '/home/zzw/data/NWPU-MOC/gt/A2_2020_orth25_27_31_3.npz'
    annotation_path = '/home/zzw/data/NWPU-MOC/gt/A0_2020_orth25_7_6_3.npz'
    # MCNN
    # model_path = '/home/zzw/data/pretrain/MCNN/all_ep_153_cls_avg_mae_8.8_cls_avg_mse_19.5_cls_weight_mse_91.4.pth'
    # MCC
    model_path = '/home/zzw/code/HKINet-NWPU/exp/05-07_21-38_MOC_RS_MCC_loss_Mse_loss_alpha_1_base_lr_5e-05_NIR=False/all_ep_533_cls_avg_mae_6.6_cls_avg_mse_19.1_cls_weight_mse_51.0.pth'

    # HKINet
    # model_path  = '/home/zzw/code/HKINet-NWPU/exp/04-06_23-59_MOC_RS_HKINet_loss_MRCloss_alpha_1_base_lr_5e-05_NIR=False/all_ep_185_cls_avg_mae_5.6_cls_avg_mse_11.3_cls_weight_mse_52.3.pth'
    # MAFA-Net
    # model_path  = '/home/zzw/data/pretrain/all_ep_377_cls_avg_mae_4.8_cls_avg_mse_10.7_cls_weight_mse_40.9.pth'
    # MOCN
    # model_path = '/home/zzw/data/pretrain/FFPN_ACA_NWPU-MOC_best.pth'

    save_dir    =  '/home/zzw/code/HKINet-NWPU'
    save_name   = 'MCC'

    # 加载真实密度
    # print(gt_maps.shape) 1024 1024 7
    gt_maps = np.load(annotation_path)['arr_0']  # 假设形状 (H, W, C)
    gt_maps = gt_maps.transpose(2,0,1)          # 变成 (C, H, W)
    print("gt_maps[0,:,:].sum()", gt_maps[0,:,:].sum())
    print("gt_maps[1,:,:].sum()",gt_maps[1,:,:].sum())
    print("gt_maps[2,:,:].sum()",gt_maps[2,:,:].sum())
    print("gt_maps[3,:,:].sum()",gt_maps[3,:,:].sum())
    print("gt_maps[4,:,:].sum()",gt_maps[4,:,:].sum())
    print("gt_maps[5,:,:].sum()",gt_maps[5,:,:].sum())
    
    # 加载模型并生成预测密度
    model = load_model(model_path, device)
    img, pred_maps, channel_sums = generate_density_maps(model, image_path, device)
    print("模型输出的预测密度图形状:", pred_maps.shape)
    
    scale_factor = 0.25
    new_size = (int(1024 * scale_factor), int(1024 * scale_factor))
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    gauss_map = smooth_gt_opencv(gt_maps, ksize=181, sigma=30)
    # pred_maps = smooth_gt_opencv(pred_maps, ksize=121, sigma=20)

    gt_maps = np.stack([
       cv2.resize(gauss_map[i], new_size, interpolation=cv2.INTER_LINEAR)
    for i in range(gt_maps.shape[0])
    ], axis=0)

    # new_size_1 = (int(1024), int(1024))
    new_size_1 = (int(1024 * scale_factor), int(1024 * scale_factor))

   
    # new_size_1 = (int(256), int(256))

    # pred_maps = torch.from_numpy(pred_maps).float()  
    # pred_maps = pred_maps.unsqueeze(0)         
    # inp = pred_maps.unsqueeze(0).to(device)
    # pred_maps = F.interpolate(
    #     pred_maps,
    #     scale_factor=8,
    #     mode='bilinear',
    #     align_corners=False,   # 建议 False，更符合像素面积对齐逻辑:contentReference[oaicite:0]{index=0}
    #     antialias=True         # 开启抗锯齿，减少插值时的高频抖动:contentReference[oaicite:1]{index=1}
    # )
    # pred_maps = pred_maps.squeeze(0).cpu().numpy() 
    pred_maps = np.stack([
        cv2.resize(pred_maps[i], new_size_1, interpolation=cv2.INTER_LINEAR)
    for i in range(pred_maps.shape[0])
    ], axis=0)
    

    print("上采样后的预测密度图形状:", pred_maps.shape)

    # pred_maps = (pred_maps > 0.01).astype(np.float32)
   

    # pred_maps: (C, H, W)  →  tmp: (H, W, C)
    # tmp = np.transpose(pred_maps, (1, 2, 0))       

    # 只要通道数不超过 4，就可以当作彩色图来处理
    # 例如 6 通道也能调用 GaussianBlur
    # tmp_blurred = cv2.GaussianBlur(tmp, ksize=(5,5), sigmaX=1, sigmaY=1)

    # 再转回 (C, H, W)
    # pred_maps = np.transpose(tmp_blurred, (2, 0, 1))



    # 绘图并保存
    plot_with_colorbars(img, gt_maps, pred_maps, save_dir, save_name, channel_sums)


    # import matplotlib.pyplot as plt

    # fig, ax = plt.subplots()
    # plt.axis('off')
    # width = 1024
    # height = 1024,
    # fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    # plt.margins(0, 0)

    # ax.imshow(target, aspect='equal')
    # plt.savefig(os.path.join(save_path, '_Target'+'.png'), dpi=300)
    # ax.imshow(output, aspect='equal')
    # plt.savefig(os.path.join(save_path, + '_pre'+'.png'), dpi=300)
    # plt.close('all')
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# import cv2
# from models.counters.HKINet import HKINet

# device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # 加载原图
# def load_original_image(image_path):
#     return cv2.imread(image_path)

# # 加载模型
# def load_model(model_path):
#     # 假设你的模型定义为MyModel
#     model = HKINet().to(device)
#     ckpt  = torch.load(model_path, map_location=device)
#     state = ckpt.get('state_dict', ckpt)
#     state = {k.replace('module.', ''): v for k,v in state.items()}
#     model.load_state_dict(state, strict=False)
#     model.eval()

#     # model = HKINet()
#     # model = torch.load(model_path)
#     # model.eval()
#     return model

# # 生成热力图
# def generate_heatmap(array):
#     # 将数组归一化到0-1范围
#     normalized = (array - array.min()) / (array.max() - array.min())
#     # 转换为热力图
#     heatmap = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
#     return heatmap

# # 主函数
# def main():
#     # 文件路径（替换为你的实际路径）
#     # img_path     = '/home/zzw/data/NWPU-MOC/rgb/A0_2020_orth25_7_6_3.png'
# # gt_npz_path  = '/home/zzw/data/NWPU-MOC/gt/A0_2020_orth25_7_6_3.npz'
#     original_image_path = '/home/zzw/data/NWPU-MOC/rgb/A0_2020_orth25_7_6_3.png'
#     annotation_path = '/home/zzw/data/NWPU-MOC/gt/A0_2020_orth25_7_6_3.npz'
#     model_path = '/home/zzw/code/HKINet-NWPU/exp/04-06_23-59_MOC_RS_HKINet_loss_MRCloss_alpha_1_base_lr_5e-05_NIR=False/all_ep_185_cls_avg_mae_5.6_cls_avg_mse_11.3_cls_weight_mse_52.3.pth'
#     output_dir = '/home/zzw/code/HKINet-NWPU/'
#     # 加载原图
#     original_image = load_original_image(original_image_path)

#     # 加载标注文件
#     annotations = np.load(annotation_path)['arr_0']  # 假设键为'arr_0'

#     # 加载模型并进行推理
#     model = load_model(model_path)
#     input_tensor = torch.from_numpy(original_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
#     input_tensor = input_tensor.to(device);
#     with torch.no_grad():
#         output = model(input_tensor)  # 模型输出形状假设为[1, 6, 1024, 1024]
#     output_maps = output.squeeze(0).cpu().numpy()  # 移除批次维度

#     # 创建一个两行七列的子图布局
#     fig, axs = plt.subplots(2, 7, figsize=(20, 5))

#     # 第一行：原图的各个通道热力图
#     for i in range(6):
#         ax = axs[0, i+1]  # 第一行从第二列开始
#         if i < annotations.shape[2]:
#             annotation_map = annotations[:, :, i]
#             heatmap = generate_heatmap(annotation_map)
#             ax.imshow(heatmap)
#             ax.set_title(f'Annotation Channel {i+1}')
#             ax.axis('off')
#         else:
#             ax.axis('off')

#     # 第二行：模型输出的各个通道热力图
#     for i in range(6):
#         ax = axs[1, i+1]  # 第二行从第二列开始
#         if i < output_maps.shape[0]:
#             output_map = output_maps[i, :, :]
#             heatmap = generate_heatmap(output_map)
#             sum_value = np.sum(output_map)
#             # 在热力图上标注数值
#             ax.imshow(heatmap)
#             ax.text(10, 20, f'Sum: {sum_value:.2f}', color='white', fontsize=8)
#             ax.set_title(f'Model Output Channel {i+1}')
#             ax.axis('off')
#         else:
#             ax.axis('off')

#     # 第一列：原图
#     axs[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
#     axs[0, 0].set_title('Original Image')
#     axs[0, 0].axis('off')

#     # 调整子图间距
#     plt.tight_layout()

#     # 保存最终结果
#     plt.savefig(f'{output_dir}/combined_result.png', dpi=100)
#     plt.close()

# if __name__ == "__main__":
#     main()





# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from torchvision import transforms
# from PIL import Image
# from models.counters.HKINet import HKINet

# # -------- 参数 --------
# img_path     = '/home/zzw/data/NWPU-MOC/rgb/A0_2020_orth25_7_6_3.png'
# gt_npz_path  = '/home/zzw/data/NWPU-MOC/gt/A0_2020_orth25_7_6_3.npz'
# # model_weights= '/home/zzw/code/HKINet-NWPU/exp/.../best.pth'
# model_weights= '/home/zzw/code/HKINet-NWPU/exp/04-06_23-59_MOC_RS_HKINet_loss_MRCloss_alpha_1_base_lr_5e-05_NIR=False/all_ep_185_cls_avg_mae_5.6_cls_avg_mse_11.3_cls_weight_mse_52.3.pth'
# device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# n_channels   = 6
# img_size     = 1024

# # -------- 加载模型 --------
# model = HKINet().to(device)
# ckpt  = torch.load(model_weights, map_location=device)
# state = ckpt.get('state_dict', ckpt)
# state = {k.replace('module.', ''): v for k,v in state.items()}
# model.load_state_dict(state, strict=False)
# model.eval()

# # -------- 加载原图 & 标注 --------
# img = Image.open(img_path).convert('RGB').resize((img_size, img_size))
# to_tensor = transforms.ToTensor()
# x   = to_tensor(img).unsqueeze(0).to(device)  # [1,3,1024,1024]
# gt  = np.load(gt_npz_path)['arr_0']            # [1024,1024,6]
# gt  = gt.transpose(2,0,1)                     # -> [6,1024,1024]

# # -------- 推理 --------
# with torch.no_grad():
#     pred = model(x).squeeze(0).cpu().numpy()  # [6,1024,1024]

# # -------- 可视化布局 --------
# fig, axes = plt.subplots(2, 7, figsize=(21, 6), dpi=100)
# for ax in axes.flatten():
#     ax.axis('off')

# # (0,0): 原图
# axes[0,0].imshow(img)
# axes[0,0].set_title('原图')

# # (1,0): 标注热力图（二值）
# im_gt = axes[1,0].imshow(gt[0], cmap='gray', vmin=0, vmax=1)
# axes[1,0].set_title('GT (示例通道)')
# fig.colorbar(im_gt, ax=axes[1,0], fraction=0.046, pad=0.04)

# # 依次填充 6 通道预测：每列两张
# for c in range(n_channels):
#     col = 1 + c // 2       # 通道 0,1 → col=1；2,3→2；4,5→3
#     row = c % 2            # 偶数通道放在 row=0，奇数放 row=1
#     ax = axes[row, col]
#     im = ax.imshow(pred[c], cmap='jet')
#     ax.set_title(f'Pred Ch{c+1}')
#     # colorbar & 积分标注
#     cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#     cbar.ax.tick_params(labelsize=8)
#     integral = pred[c].sum()
#     ax.text(0.98, 0.02, f'Sum={integral:.2f}',
#             transform=ax.transAxes, ha='right', va='bottom',
#             color='white', fontsize=8,
#             bbox=dict(facecolor='black', alpha=0.5, pad=2))

# plt.tight_layout()
# plt.savefig('figure2_layout.png', bbox_inches='tight')
# plt.show()
