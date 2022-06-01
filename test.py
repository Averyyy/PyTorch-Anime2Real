#!/usr/bin/python3

import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from models import Generator, Discriminator
from datasets import ImageDataset
from utils import tensor2image

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='output/netG_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='output/netG_B2A.pth', help='B2A generator checkpoint file')
parser.add_argument('--dis_A', type=str, default='output/netD_A.pth', help='B2A generator checkpoint file')
parser.add_argument('--dis_B', type=str, default='output/netD_B.pth', help='B2A generator checkpoint file')

opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()

# Load state dicts
netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
netG_B2A.load_state_dict(torch.load(opt.generator_B2A))
netD_A.load_state_dict(torch.load(opt.dis_A))
netD_B.load_state_dict(torch.load(opt.dis_B))

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()
netD_A.eval()
netD_B.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)


# Dataset loader
transforms_ = [ transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test'), 
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
###################################

###### Testing######

# Create output dirs if they don't exist
if not os.path.exists('output/A'):
    os.makedirs('output/A')
if not os.path.exists('output/B'):
    os.makedirs('output/B')
if not os.path.exists('output/losses'):
    os.makedirs('output/losses')

for i, batch in enumerate(dataloader):
    # Set model input
    real_A = Variable(input_A.copy_(batch['A']))
    real_B = Variable(input_B.copy_(batch['B']))

    # # Generate output
    # fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
    # fake_A = 0.5*(netG_B2A(real_B).data + 1.0)

    # # Save image files
    # save_image(real_A, f'output/A/real_{i+1}.png')
    # save_image(fake_B, f'output/A/fake{i+1}.png')
    # save_image(real_B, f'output/B/real_{i+1}.png')
    # save_image(fake_A, f'output/B/fake{i+1}.png')

    # Save losses
    # Identity loss
    # G_A2B(B) should equal B if real B is fed
    same_B = netG_A2B(real_B)
    loss_identity_B = criterion_identity(same_B, real_B)*5.0
    # G_B2A(A) should equal A if real A is fed
    same_A = netG_B2A(real_A)
    loss_identity_A = criterion_identity(same_A, real_A)*5.0

    # GAN loss
    fake_B = netG_A2B(real_A)
    # pred_fake = netD_B(fake_B)
    # loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

    fake_A = netG_B2A(real_B)
    # pred_fake = netD_A(fake_A)
    # loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

    # # Cycle loss
    recovered_A = netG_B2A(fake_B)
    loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

    recovered_B = netG_A2B(fake_A)
    loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

    # # Total loss
    # loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
    with open(f'output/losses/loss_{i+1}.txt', 'w') as f:
        f.write(f'Identity loss A: {loss_identity_A.item()} \nIdentity loss B: {loss_identity_B.item()}\nCycle loss ABA: {loss_cycle_ABA.item()} \nCycle loss BAB: {loss_cycle_BAB.item()}')
    fake_B = 0.5*(fake_B.data + 1.0)
    fake_A = 0.5*(fake_A.data + 1.0)
    real_A = 0.5*(real_A.data + 1.0)
    real_B = 0.5*(real_B.data + 1.0)
    save_image(real_A, f'output/A/{i+1}_real.png')
    save_image(fake_B, f'output/A/{i+1}_fake.png')
    save_image(real_B, f'output/B/{i+1}_real.png')
    save_image(fake_A, f'output/B/{i+1}_fake.png')
    
    
    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

sys.stdout.write('\n')
###################################
