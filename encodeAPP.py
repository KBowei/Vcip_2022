import argparse
import json
import os
import time
import torch
from torch.autograd import Variable
import arithmetic_coding as ac
from Encode import enc_lossy


parser = argparse.ArgumentParser(description='VCIP 2022', conflict_handler='resolve')
parser.add_argument('--cfg_file', type=str, default='./encode.cfg')
cfg_args, unknown = parser.parse_known_args()

cfg_file = cfg_args.cfg_file
with open(cfg_file, 'r') as f:
    cfg_dict = json.load(f)
    
    for key, value in cfg_dict.items():
        if isinstance(value, int):
            parser.add_argument('--{}'.format(key), type=int, default=value)
        elif isinstance(value, float):
            parser.add_argument('--{}'.format(key), type=float, default=value)
        else:
            parser.add_argument('--{}'.format(key), type=str, default=value)

cfg_args, unknown = parser.parse_known_args()

# parameters
parser.add_argument('--bin_dir', type=str, default=cfg_args.bin_dir)
parser.add_argument('--recon_dir', type=str, default=cfg_args.recon_dir)
parser.add_argument('--log_dir', type=str, default=cfg_args.log_dir)
parser.add_argument('--input_dir', type=str, default=cfg_args.input_dir)
parser.add_argument('--model_path', type=str, default=cfg_args.model_path)

parser.add_argument('--model_qp', type=int, default=cfg_args.model_qp)
parser.add_argument('--qp_shift', type=int, default=cfg_args.qp_shift)
parser.add_argument('--code_block_size', type=int, default=cfg_args.code_block_size)

def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def dec_binary(dec, bin_num):
    value = 0
    freqs = ac.SimpleFrequencyTable([1, 1])
    for i in range(bin_num):
        dec_c = dec.read(freqs)
        value = value + (2**(bin_num-1-i))*dec_c
    return value


def main():
    args = parser.parse_args()

    if not os.path.exists(args.bin_dir):
        os.makedirs(args.bin_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.recon_dir):
        os.makedirs(args.recon_dir)

    logfile = open(os.path.join(args.log_dir, 'enc_log_{}.txt'.format(args.img_name[0:-4])), 'a')

    start = time.time()

    bin_name = args.img_name[0:-4] + '_' + str(args.model_qp) + '_' + str(args.qp_shift)
    bit_out = ac.CountingBitOutputStream(
        bit_out=ac.BitOutputStream(open(os.path.join(args.bin_dir, bin_name + '.bin'), "wb")))
    enc = ac.ArithmeticEncoder(bit_out)
    freqs_resolution = 1e7
    
    # Encode
    height, width, psnr = enc_lossy(args, bin_name, enc, freqs_resolution, logfile)

    bit_out.close()
    print('bit_out closed!')
    logfile.write('bit_out closed!' + '\n')
    
    end = time.time()
    print('encoding finished!')
    logfile.write('encoding finished!' + '\n')
    print('encoding time: ', end - start)
    logfile.write('encoding time: ' + str(end - start) + '\n')

    filesize = bit_out.num_bits / height / width
    print('BPP: ', filesize)
    logfile.write('BPP: ' + str(filesize) + '\n')
    logfile.flush()

    print('PSNR: ', psnr)
    logfile.write('PSNR: ' + str(psnr) + '\n')
    logfile.flush()
    logfile.close()

if __name__ == "__main__":
    main()
