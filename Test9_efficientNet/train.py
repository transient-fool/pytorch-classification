import os
import math
import argparse
import sys
import torch.nn as nn

import pywt
import scipy
import torch
import torch.optim as optim
from PIL import Image
from scipy.ndimage import median_filter
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import torch.nn.functional as F
from model import efficientnet_b0 as create_model
from my_dataset import MyDataSet
# from torchvision.transforms import NonLocalMeansDenoise
# from torchvision.transforms import AddGaussianNoise
from tqdm import tqdm

from utils import read_split_data, train_one_epoch, evaluate


# def wavelet_denoise(image):
#     # Wavelet transform of image, thresholding, and inverse wavelet transform
#     coeffs = pywt.dwt2(image.numpy(), 'haar')  # Applying 2D Haar wavelet transform
#     coeffs = tuple(map(lambda x: pywt.threshold(x, value=0.5 * np.std(x), mode='soft'), coeffs))  # Thresholding
#     denoised_image = pywt.idwt2(coeffs, 'haar')  # Inverse wavelet transform
#     return denoised_image

import torch
import torch.nn.functional as F


def total_variation_denoise(image, tv_weight=0.1, num_iter=5, lr=0.1):
    # Convert input image to a PyTorch tensor with gradients enabled
    img_var = image.clone().detach().requires_grad_(True)

    # Use Adam optimizer to optimize the image tensor
    optimizer = torch.optim.Adam([img_var], lr=lr)

    def tv_loss(img):
        # Function to calculate total variation loss
        img = F.pad(img, (1, 1, 1, 1), mode='constant', value=0)
        dy = img[:, :, 2:, :] - img[:, :, :-2, :]  # Calculate vertical differences
        dx = img[:, :, :, 2:] - img[:, :, :, :-2]  # Calculate horizontal differences
        return torch.sum(torch.abs(dx)) + torch.sum(torch.abs(dy))

    for _ in range(num_iter):
        optimizer.zero_grad()
        loss = tv_loss(img_var)
        loss.backward()
        optimizer.step()

    # Detach the tensor from computation graph and return as numpy array
    return img_var.detach().numpy()


def wavelet_denoise_transform(image):
    # 将输入图像转换为numpy数组
    img_array = image.numpy()  # 将PyTorch张量转换为NumPy数组

    # 小波变换
    coeffs = pywt.dwt2(img_array, 'haar')  # 使用'haar'小波基进行二维小波变换

    # 小波系数去噪处理，可以根据实际需求调整阈值或其他参数
    coeffs = tuple(map(lambda x: pywt.threshold(x, value=0.3 * np.max(x), mode='soft'), coeffs))

    # 逆小波变换
    denoised_img = pywt.idwt2(coeffs, 'db4')  # 使用'haar'小波基进行二维小波逆变换

    # 将处理后的图像转换回PyTorch张量
    denoised_img = torch.from_numpy(denoised_img).float()  # 转换为PyTorch张量

    return denoised_img
import torchvision.transforms as transforms


def median_filter_transform(tensor_image):
    # Check if input is a tensor
    if isinstance(tensor_image, torch.Tensor):
        # Convert tensor to numpy array
        image_np = tensor_image.numpy().transpose((1, 2, 0))  # Convert CHW to HWC

        # Apply median filter
        filtered_np = median_filter(image_np, size=2)  # Adjust size as needed

        # Convert numpy array back to tensor
        filtered_tensor = torch.from_numpy(filtered_np.transpose((2, 0, 1)))  # Convert HWC to CHW

        return filtered_tensor
    else:
        raise TypeError(f"Expected torch.Tensor, got {type(tensor_image)}")


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")
   # 训练集和验证集
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    num_model = "B0"

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size[num_model]),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                     # NonLocalMeansDenoise(h=10, templateWindowSize=7, searchWindowSize=21)
                                     # transforms.Lambda(wavelet_denoise_transform),
                                     # AddGaussianNoise(mean=0, std=0.1)
                                     # transforms.Lambda(total_variation_denoise),
                                     # transforms.Lambda(median_filter_transform)
                                     ]),
        "val": transforms.Compose([transforms.Resize(img_size[num_model]),
                                   transforms.CenterCrop(img_size[num_model]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                   # NonLocalMeansDenoise(h=10, templateWindowSize=7, searchWindowSize=21)
                                   # transforms.Lambda(wavelet_denoise_transform),
                                   # AddGaussianNoise(mean=0, std=0.1)
                                   # transforms.Lambda(total_variation_denoise),
                                   # transforms.Lambda(median_filter_transform)
                                   ])}

    # 实例化训练数据集,这里为什么需要一个单独的类来装自己验证集和训练集呢
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size # 以命令行的方式装入
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 1])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn) # 这里加载数据集的时候就已经设置好了batch_size吗缘来

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    #如果存在预训练权重则载入
    model = create_model(num_classes=args.num_classes).to(device)
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # k是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后一个卷积层和全连接层外，其他权重全部冻结
            if ("features.top" not in name) and ("classifier" not in name):
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    #efficientnet原来的参数
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.Adam(params, lr=0.0001)
    # loss_function = nn.CrossEntropyLoss() # resnet的损失函数
    # val_num = len(val_dataset)
    # train_steps = len(train_loader)



    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    for epoch in range(args.epochs):
        # train
        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()

        # validate
        # acc = evaluate(model=model,
        #                data_loader=val_loader,
        #                device=device)
        # print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))

        acc, recall, f1 = evaluate(model=model,
                                   data_loader=val_loader,
                                   device=device)
        print("[epoch {}] accuracy: {:.3f}, recall: {:.3f}, F1: {:.3f}".format(epoch, acc, recall, f1))
        #
        tags = ["loss", "accuracy", "recall", "f1", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], recall, epoch)
        tb_writer.add_scalar(tags[3], f1, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="D:/fishvideo/classification")

    # download model weights
    # 链接: https://pan.baidu.com/s/1ouX0UmjCsmSx3ZrqXbowjw  密码: 090iD:\pyproject0\deep-learning-for-image-processing-master\pytorch_classification\Test9_efficientNet\efficientnetb0.pth
    parser.add_argument('--weights', type=str, default="",
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
