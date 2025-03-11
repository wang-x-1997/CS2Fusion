# coding=utf-8
from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import kornia
import matplotlib.pyplot as plt

from dataset import fusiondata
import net
from TD import TD
from loss import SimMaxLoss, SimMinLoss


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    # 训练设置
    parser = argparse.ArgumentParser(description='Image Fusion Network Implementation')
    parser.add_argument('--dataset', type=str, default='data', help='dataset name')
    parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
    parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
    parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--input_nc', type=int, default=1, help='input image channels')
    parser.add_argument('--output_nc', type=int, default=1, help='output image channels')
    parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning Rate. Default=0.002')
    parser.add_argument('--lr1', type=float, default=0.0002, help='Learning Rate. Default=0.002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--lamb', type=int, default=150, help='weight on L1 term in objective')
    parser.add_argument('--alpha', type=float, default=0.25, help='alpha parameter for loss')
    parser.add_argument('--ema_decay', type=float, default=0.9, help='ema_decay')
    parser.add_argument('--checkpoint_dir', type=str, default='./', help='directory to save checkpoints')
    parser.add_argument('--pretrained_ae', type=str, default=None, help='path to pretrained autoencoder model')
    return parser.parse_args()


def setup_environment(opt):
    use_cuda = not opt.cuda and torch.cuda.is_available()

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def load_data(opt):
    print('===> Loading datasets')
    root_path = "data/"
    dataset = fusiondata(os.path.join(root_path, opt.dataset))
    training_data_loader = DataLoader(
        dataset=dataset,
        num_workers=opt.threads,
        batch_size=opt.batchSize,
        shuffle=True
    )
    return training_data_loader


def build_models(opt, device):
    print('===> Building models')

    # 初始化融合网络
    model_DE = net.Fusion_strage()
    model_DE = model_DE.to(device)

    # 加载预训练的自编码器模型或初始化新的
    if opt.pretrained_ae:
        print(f'Loading pretrained autoencoder from {opt.pretrained_ae}')
        model_AE = torch.load(opt.pretrained_ae)
    else:
        model_AE = net.AutoEncoder()
        model_AE = model_AE.to(device)

    print('---------- Networks initialized -------------')

    return model_AE, model_DE


def setup_optimizers(opt, model_AE, model_DE):
    optimizer_AE = optim.Adam(model_AE.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    optimizer_DE = optim.Adam(model_DE.parameters(), lr=opt.lr1, betas=(0.9, 0.999))

    scheduler_AE = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_AE, [5, 10, 15], gamma=0.1)
    scheduler_DE = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_DE, [5, 10, 15], gamma=0.1)

    return optimizer_AE, optimizer_DE, scheduler_AE, scheduler_DE


def setup_loss_functions(opt, device):
    # 初始化损失函数
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    ssim_loss = kornia.losses.SSIMLoss(3, reduction='mean')
    td_loss = TD(device=device)

    # 对比损失
    contrast_loss = [
        SimMaxLoss(metric='cos', alpha=opt.alpha).to(device),
        SimMinLoss(metric='cos').to(device),
        SimMaxLoss(metric='cos', alpha=opt.alpha).to(device)
    ]

    return {
        'mse': mse_loss,
        'l1': l1_loss,
        'ssim': ssim_loss,
        'td': td_loss,
        'contrast': contrast_loss
    }


def save_checkpoint(epoch, model_AE, model_DE, prefix, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    ae_path = os.path.join(checkpoint_dir, f"{prefix}_AE_epoch_{epoch}.pth")
    de_path = os.path.join(checkpoint_dir, f"{prefix}_DE_epoch_{epoch}.pth")

    torch.save(model_AE, ae_path)
    torch.save(model_DE, de_path)
    print(f"Checkpoint saved to {checkpoint_dir}")


def train_epoch(epoch, k, model_AE, model_DE, training_data_loader,
                criterion, optimizer_AE, optimizer_DE, device):
    model_AE.train()
    model_DE.train()

    loss_ae_values = []
    loss_fusion_values = []
    loss_contrast_values = []

    for iteration, batch in enumerate(training_data_loader, 1):
        imgA, imgB, imgC = batch[0].to(device), batch[1].to(device), batch[2].to(device)

        E1, E2, E3, E4, f, F = model_AE(imgA, imgB)

        e1 = [E1[0].detach(), E1[1].detach()]
        e2 = [E2[0].detach(), E2[1].detach()]
        e3 = [E3[0].detach(), E3[1].detach()]
        e4 = [E4[0].detach(), E4[1].detach()]
        ef = f.detach()

        FM, M, mf, W = model_DE(e1, e2, e3, e4, ef)

        AE1, AE2, AE3, AE4, Af, AF = model_AE(imgA, imgC)

        Ae1 = [AE1[0].detach(), AE1[1].detach()]
        Ae2 = [AE2[0].detach(), AE2[1].detach()]
        Ae3 = [AE3[0].detach(), AE3[1].detach()]
        Ae4 = [AE4[0].detach(), AE4[1].detach()]
        Aef = Af.detach()

        AFM, AM, Amf, AW = model_DE(Ae1, Ae2, Ae3, Ae4, Aef)

        loss_ae = torch.norm(F * W[0] - imgA * W[0]) + torch.norm(F * W[1] - imgB * W[1]) + \
                  criterion['ssim'](F, imgA) + criterion['ssim'](F, imgB)

        loss_contrast_1 = (criterion['contrast'][0](M[0], mf[0]) +
                           criterion['contrast'][0](M[1], mf[1])) + \
                          criterion['contrast'][1](mf[1], mf[0])

        loss_contrast_2 = criterion['contrast'][0](mf[0], AM[0]) + \
                          criterion['contrast'][1](mf[0], AM[1])

        loss_contrast = loss_contrast_1 + k * loss_contrast_2

        optimizer_AE.zero_grad()
        loss_ae.backward(retain_graph=True)
        optimizer_AE.step()

        optimizer_DE.zero_grad()
        loss_contrast.backward()
        optimizer_DE.step()

        loss_ae_values.append(loss_ae.item())
        loss_contrast_values.append(loss_contrast_1.item())
        loss_fusion_values.append(loss_contrast_2.item())

        print(f'Epoch {epoch} Batch {iteration}: AE Loss: {loss_ae.item():.4f}, '
              f'Contrast Loss: {loss_contrast_1.item():.4f}, Fusion Loss: {loss_contrast_2.item():.4f}')

    return {
        'ae': sum(loss_ae_values) / len(loss_ae_values),
        'contrast': sum(loss_contrast_values) / len(loss_contrast_values),
        'fusion': sum(loss_fusion_values) / len(loss_fusion_values)
    }


def plot_losses(losses_history, save_path=None):
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(losses_history['ae']) + 1)

    plt.subplot(3, 1, 1)
    plt.plot(epochs, losses_history['ae'], 'b-', label='AE Loss')
    plt.title('Autoencoder Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(epochs, losses_history['contrast'], 'r-', label='Contrast Loss')
    plt.title('Contrast Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(epochs, losses_history['fusion'], 'g-', label='Fusion Loss')
    plt.title('Fusion Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def main():
    opt = parse_args()

    device = setup_environment(opt)

    training_data_loader = load_data(opt)

    model_AE, model_DE = build_models(opt, device)

    optimizer_AE, optimizer_DE, scheduler_AE, scheduler_DE = setup_optimizers(opt, model_AE, model_DE)

    criterion = setup_loss_functions(opt, device)

    k_values = [2] # lambda
    max_epochs = 20

    losses_history = {'ae': [], 'contrast': [], 'fusion': []}

    for k in k_values:
        print(f"Training with k={k}")

        for epoch in range(1, max_epochs + 1):
            print(f"Starting epoch {epoch}/{max_epochs}")

            epoch_losses = train_epoch(
                epoch, k, model_AE, model_DE,
                training_data_loader, criterion,
                optimizer_AE, optimizer_DE, device
            )


            scheduler_AE.step()
            scheduler_DE.step()


            for key in losses_history:
                losses_history[key].append(epoch_losses[key])


            if epoch % 20 == 0:
                save_checkpoint(
                    epoch, model_AE, model_DE,
                    f"k{k}", opt.checkpoint_dir
                )

            print(f"Completed epoch {epoch}/{max_epochs}")

    # loss curve
    plot_losses(losses_history, save_path=os.path.join(opt.checkpoint_dir, "loss_plot.png"))


if __name__ == '__main__':
    main()