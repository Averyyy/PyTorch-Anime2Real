#!/usr/bin/python3

import argparse
import itertools

import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from torch.utils import save_image
from torch.autograd import Variable
from PIL import Image
import torch
import numpy as np

from models import Generator
from model_r2a2r import Discriminator_A2R, Generator_A2R
# from utils import Logger
from utils import weights_init_normal
from datasets import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--input_nc', type=int, default=3,
                    help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3,
                    help='number of channels of output data')
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batchSize", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument('--dataroot', type=str, default='datasets/a2r/',
                    help='root directory of the dataset')
parser.add_argument('--cuda', action='store_true', default=True, help='use GPU computation')
parser.add_argument('--size', type=int, default=256,
                    help='size of the data crop (squared assumed)')

opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks

netG_R2A = Generator(opt.output_nc, opt.input_nc, n_residual_blocks=9)
netG_A2R = Generator_A2R()
netD_A2R = Discriminator_A2R()

if opt.cuda:
    netG_R2A.cuda()
    netG_A2R.cuda()
    netD_A2R.cuda()

# if opt.mps:
#     netG_R2A.to(torch.device('mps'))
#     netG_A2R.to(torch.device('mps'))
#     netD_A2R.to(torch.device('mps'))

# check if r2a exists
if os.path.exists('models/pretrained/netG_B2A.pth'):
    netG_R2A.load_state_dict(torch.load('models/pretrained/netG_B2A.pth'))
else:
    print('No r2a pretrained model found!')
    exit()
print('--------------Initialize models')
# netG_R2A.eval()

# Lossess
criterion_identity = torch.nn.L1Loss()
adversarial_loss = torch.nn.BCELoss()

# Dataset loader
transforms_ = [transforms.Resize(int(opt.size*1.12), Image.BICUBIC),
               transforms.RandomCrop(opt.size),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

optimizer_G = torch.optim.Adam(netG_A2R.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(netD_A2R.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor
input_Real = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)


###################################

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    print("Epoch: %d" % epoch)
    for i, batch in enumerate(dataloader):
        # Set model input
        # real_A = Variable(input_A.copy_(batch['A']))
        real_img = Variable(input_Real.copy_(batch['B']))
        generated_anime = netG_R2A(real_img)

        # train the generator
        optimizer_G.zero_grad()
        # z = Variable(Tensor(np.random.normal(0, 1, (batch['B'].shape[0], opt.latent_dim))))
        gen_img = netG_A2R(generated_anime)
        loss_ad = adversarial_loss(Discriminator_A2R(gen_img), target_real)
        loss_iden = criterion_identity(gen_img, real_img)

        g_loss = loss_ad + loss_iden*5.0
        g_loss.backward()

        optimizer_G.step()


        # train the discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(Discriminator_A2R(real_img), target_real)
        fake_loss = adversarial_loss(Discriminator_A2R(gen_img.detach()), target_fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        if (i % 20 == 0):
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D Loss: %f] [G loss: %f]"
                % (epoch, opt.n_epoches, i, len(dataloader), d_loss.item(), g_loss.item())
            )

        batches_done = epoch * len(dataloader) + i
        # if batches_done % opt.sample_interval == 0:
        #     save_image(gen_img.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)




    if not os.path.exists('output/checkpoints'):
        os.makedirs('output/checkpoints')
    if epoch % 20 == 0:
        # Save models checkpoints
        torch.save(netG_A2R.state_dict(),
                   f'output/checkpoints/netG_A2B_epoch_{epoch}.pth')
        torch.save(netD_A2R.state_dict(),
                   f'output/checkpoints/netG_B2A_epoch_{epoch}.pth')
        # save current checkpoint
        torch.save(netG_A2R.state_dict(), 'output/netG_A2B.pth')
        torch.save(netD_A2R.state_dict(), 'output/netG_B2A.pth')
###################################
