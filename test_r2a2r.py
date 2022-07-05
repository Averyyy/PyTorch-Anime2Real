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
parser.add_argument('--generator_A2B', type=str, default='output/netG_A2R.pth', help='A2B generator checkpoint file')
parser.add_argument('--dis_A', type=str, default='output/netD_A2R.pth', help='B2A generator checkpoint file')

if __name__ == "__main__":
    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    ###### Definition of variables ######
    # Networks
    netG_A2R = Generator(opt.input_nc, opt.output_nc, n_residual_blocks=6)
    netD_A = Discriminator(opt.input_nc)

    # Lossess
    adversarial_loss = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    if opt.cuda:
        netG_A2R.cuda()
        netD_A.cuda()

    # Load state dicts
    netG_A2R.load_state_dict(torch.load(opt.generator_A2B))
    netD_A.load_state_dict(torch.load(opt.dis_A))

    # Set model's test mode
    netG_A2R.eval()
    netD_A.eval()

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
    if not os.path.exists('output/losses'):
        os.makedirs('output/losses')

    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        generated_real = netG_A2R(real_A)

        loss_ad = adversarial_loss(netD_A(generated_real), target_real)


        # # Total loss
        with open(f'output/losses/loss.txt', 'a') as f:
            f.write(f'[Index{i}: ] [Loss: {loss_ad}]\n');
        generated_real = 0.5*(generated_real.data + 1.0)
        real_A = 0.5*(real_A.data + 1.0)
        save_image(real_A, f'output/A/{i+1}_real.png')
        save_image(generated_real, f'output/A/{i+1}_fake.png')


        sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

    sys.stdout.write('\n')
    ###################################
