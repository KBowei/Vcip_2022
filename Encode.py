import argparse
import torch
from torch.autograd import Variable
import os
from PIL import Image
import numpy as np
from torch.nn import functional as F
import copy
import arithmetic_coding as ac
from utils import img2patch, img2patch_padding, rgb2yuv, yuv2rgb, find_min_and_max, qp_shifts, model_lambdas
from Model import *
import pdb


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def write_binary(enc, value, bin_num):
    bin_v = '{0:b}'.format(value).zfill(bin_num)
    freqs = ac.SimpleFrequencyTable([1, 1])
    for i in range(bin_num):
        enc.write(freqs, int(bin_v[i]))


def enc_lossy(args, bin_name, enc, freqs_resolution, logfile):

    code_block_size = args.code_block_size
    trans_steps = 4

    init_scale = qp_shifts[args.model_qp][args.qp_shift]
    print(init_scale)
    logfile.write(str(init_scale) + '\n')
    logfile.flush()

    freqs = ac.SimpleFrequencyTable(np.ones([8], dtype=np.int))
    enc.write(freqs, args.model_qp)

    freqs = ac.SimpleFrequencyTable(np.ones([64], dtype=np.int))
    enc.write(freqs, args.qp_shift)

    # checkpoint = torch.load(os.path.join(args.model_path, 'affine_' + str(model_lambdas[args.model_qp]) + '.pth'))
    checkpoint = torch.load('/model/chenhang/Iwave_model/affine_0.0625/model_epoch040_hard.pth')
    # print(os.path.join(args.model_path, str(model_lambdas[args.model_qp]) + '_40.pth'))

    all_part_dict = checkpoint['state_dict']
    models_dict = {}
    model = Model(wavelet_affine=True).to(device)
    model.load_state_dict(all_part_dict)

    models_dict['scale_net'] = model.scale_net
    models_dict['transform0'] = model.wavelet_transform[0]
    models_dict['transform1'] = model.wavelet_transform[1]
    models_dict['transform2'] = model.wavelet_transform[2]
    models_dict['transform3'] = model.wavelet_transform[3]

    models_dict['transform'] = [models_dict['transform0'], models_dict['transform1'], models_dict['transform2'], models_dict['transform3']]

    models_dict['entropy_LL_Y'] = model.coding_LL_Y
    models_dict['entropy_LL_U'] = model.coding_LL_U
    models_dict['entropy_LL_V'] = model.coding_LL_V

    models_dict['entropy_HL_Y'] = model.coding_HL_list_Y
    models_dict['entropy_HL_U'] = model.coding_HL_list_U
    models_dict['entropy_HL_V'] = model.coding_HL_list_V

    models_dict['entropy_LH_Y'] = model.coding_LH_list_Y
    models_dict['entropy_LH_U'] = model.coding_LH_list_U
    models_dict['entropy_LH_V'] = model.coding_LH_list_V

    models_dict['entropy_HH_Y'] = model.coding_HH_list_Y
    models_dict['entropy_HH_U'] = model.coding_HH_list_U
    models_dict['entropy_HH_V'] = model.coding_HH_list_V

    models_dict['post'] = model.post
  
    print('Load pre-trained model succeed!')
    logfile.write('Load pre-trained model succeed!' + '\n')
    logfile.flush()

    img_path = os.path.join(args.input_dir, args.img_name)

    HL_list = []
    LH_list = []
    HH_list = []
    with torch.no_grad():

        logfile.write(img_path + '\n')
        logfile.flush()

        img = Image.open(img_path)
        img = np.array(img, dtype=np.float32)
        img = torch.from_numpy(img)
        # img -> [n,c,h,w]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)
        original_img = img

        # img -> (%16 == 0)
        size = img.size()
        height = size[2]
        width = size[3]
        # encode height and width, in the range of [0, 2^15=32768]
        write_binary(enc, height, 15)
        write_binary(enc, width, 15)
    
        pad_h = int(np.ceil(height / 16)) * 16 - height
        pad_w = int(np.ceil(width / 16)) * 16 - width
        paddings = (0, pad_w, 0, pad_h)
        img = F.pad(img, paddings, 'replicate')
        # img -> [1, 3, h, w]
        img = rgb2yuv(img)
        LL = to_variable(img)

        used_scale = init_scale + models_dict['scale_net']()

        for i in range(trans_steps):
            LL, HL, LH, HH = models_dict['transform'][i].forward_trans(LL)
            HL_list.append(torch.round(HL/used_scale))
            LH_list.append(torch.round(LH/used_scale))
            HH_list.append(torch.round(HH/used_scale))
        LL = torch.round(LL/used_scale)
        min_v, max_v = find_min_and_max(LL, HL_list, LH_list, HH_list)

        # for all models, the quantized coefficients are in the range of [-6016, 12032]
        # 15 bits to encode this range
        for i in range(3):
            for j in range(13):
                tmp = min_v[i, j] + 8000
                write_binary(enc, tmp, 15)
                tmp = max_v[i, j] + 8000
                write_binary(enc, tmp, 15)
                 
        yuv_low_bound = min_v.min(axis=0)
        yuv_high_bound = max_v.max(axis=0)
        shift_min = min_v - yuv_low_bound
        shift_max = max_v - yuv_low_bound

        subband_h = [(height + pad_h) // 2, (height + pad_h) // 4, (height + pad_h) // 8, (height + pad_h) // 16]
        subband_w = [(width + pad_w) // 2, (width + pad_w) // 4, (width + pad_w) // 8, (width + pad_w) // 16]

        padding_sub_h = [(int(np.ceil(tmp / code_block_size)) * code_block_size - tmp) for tmp in subband_h]
        padding_sub_w = [(int(np.ceil(tmp / code_block_size)) * code_block_size - tmp) for tmp in subband_w]

        coded_coe_num = 0
        # compress LL
        tmp_stride = subband_w[3] + padding_sub_w[3]
        tmp_hor_num = tmp_stride // code_block_size
        paddings = (0, padding_sub_w[3], 0, padding_sub_h[3])
        enc_LL = F.pad(LL, paddings, "constant")
        # [num_patch, 3, h, w]
        enc_LL = img2patch(enc_LL, code_block_size, code_block_size, code_block_size)
        paddings = (6, 6, 6, 6)
        enc_LL = F.pad(enc_LL, paddings, "constant")
        paddings = (0, padding_sub_w[3], 0, padding_sub_h[3])
        LL_context = F.pad(LL, paddings, "reflect")
        paddings = (6, 6, 6, 6)
        LL_context = F.pad(LL_context, paddings, "reflect")
        LL_context = img2patch_padding(LL_context, code_block_size + 12, code_block_size + 12, code_block_size, 6)

        for h_i in range(code_block_size):
            for w_i in range(code_block_size):
                cur_ct_y = copy.deepcopy(enc_LL[:, 0:1, h_i:h_i + 13, w_i:w_i + 13])
                cur_ct_y[:, :, 13 // 2 + 1:13, :] = 0.
                cur_ct_y[:, :, 13 // 2, 13 // 2:13] = 0.
                prob_y = models_dict['entropy_LL_Y'](cur_ct_y, yuv_low_bound[0], yuv_high_bound[0]).cpu().data.numpy()
                prob_y = (prob_y * freqs_resolution).astype(np.int64)
                index = enc_LL[:, 0, h_i + 6, w_i + 6].cpu().data.numpy().astype(np.int64)
                for sample_idx, prob_sample in enumerate(prob_y):
                    coe_id = (sample_idx // tmp_hor_num) * tmp_hor_num * code_block_size * code_block_size + \
                             h_i * tmp_stride + \
                             (sample_idx % tmp_hor_num) * code_block_size + \
                             w_i
                    if (coe_id % tmp_stride) < subband_w[3] and (coe_id // tmp_stride) < subband_h[3]:
                        # 不在padding区域内:
                        if shift_min[0, 0] < shift_max[0, 0]:
                            freqs = ac.SimpleFrequencyTable(
                                prob_sample[shift_min[0, 0]:shift_max[0, 0] + 1])
                            data = index[sample_idx] - min_v[0, 0]
                            assert data >= 0
                            enc.write(freqs, data)
                        coded_coe_num = coded_coe_num + 1                   
        for h_i in range(code_block_size):
            for w_i in range(code_block_size):
                cur_ct_u = copy.deepcopy(enc_LL[:, 1:2, h_i:h_i + 13, w_i:w_i + 13])
                cur_ct_u[:, :, 13 // 2 + 1:13, :] = 0.
                cur_ct_u[:, :, 13 // 2, 13 // 2:13] = 0.
                prob_u = models_dict['entropy_LL_U'](cur_ct_u, LL_context[:, 0:1, h_i:h_i + 13, w_i:w_i + 13],
                                                     yuv_low_bound[0], yuv_high_bound[0]).cpu().data.numpy()
                prob_u = (prob_u * freqs_resolution).astype(np.int64)
                index = enc_LL[:, 1, h_i + 6, w_i + 6].cpu().data.numpy().astype(np.int64)
                for sample_idx, prob_sample in enumerate(prob_u):
                    coe_id = (sample_idx // tmp_hor_num) * tmp_hor_num * code_block_size * code_block_size + \
                             h_i * tmp_stride + \
                             (sample_idx % tmp_hor_num) * code_block_size + \
                             w_i
                    if (coe_id % tmp_stride) < subband_w[3] and (coe_id // tmp_stride) < subband_h[3]:
                        # 不在padding区域内:
                        if shift_min[1, 0] < shift_max[1, 0]:
                            freqs = ac.SimpleFrequencyTable(
                                prob_sample[shift_min[1, 0]:shift_max[1, 0] + 1])
                            data = index[sample_idx] - min_v[1, 0]
                            assert data >= 0
                            enc.write(freqs, data)
                        coded_coe_num = coded_coe_num + 1                
        for h_i in range(code_block_size):
            for w_i in range(code_block_size):   
                cur_ct_v = copy.deepcopy(enc_LL[:, 2:3, h_i:h_i + 13, w_i:w_i + 13])
                cur_ct_v[:, :, 13 // 2 + 1:13, :] = 0.
                cur_ct_v[:, :, 13 // 2, 13 // 2:13] = 0.
                prob_v = models_dict['entropy_LL_V'](cur_ct_v, LL_context[:, 0:2, h_i:h_i + 13, w_i:w_i + 13],
                                                     yuv_low_bound[0], yuv_high_bound[0]).cpu().data.numpy()
                prob_v = (prob_v * freqs_resolution).astype(np.int64)
                index = enc_LL[:, 2, h_i + 6, w_i + 6].cpu().data.numpy().astype(np.int64)
                for sample_idx, prob_sample in enumerate(prob_v):
                    coe_id = (sample_idx // tmp_hor_num) * tmp_hor_num * code_block_size * code_block_size + \
                             h_i * tmp_stride + \
                             (sample_idx % tmp_hor_num) * code_block_size + \
                             w_i
                    if (coe_id % tmp_stride) < subband_w[3] and (coe_id // tmp_stride) < subband_h[3]:
                        # 不在padding区域内:
                        if shift_min[2, 0] < shift_max[2, 0]:
                            freqs = ac.SimpleFrequencyTable(
                                prob_sample[shift_min[2, 0]:shift_max[2, 0] + 1])
                            data = index[sample_idx] - min_v[2, 0]
                            assert data >= 0
                            enc.write(freqs, data)
                        coded_coe_num = coded_coe_num + 1    
        print('LL encoded...')
        LL = LL * used_scale

        for i in range(trans_steps):
            j = trans_steps - 1 - i
            tmp_stride = subband_w[j] + padding_sub_w[j]
            tmp_hor_num = tmp_stride // code_block_size
            patch_num = tmp_hor_num*tmp_hor_num
            # compress HL
            paddings = (0, padding_sub_w[j], 0, padding_sub_h[j])
            enc_oth = F.pad(HL_list[j], paddings, "constant")
            enc_oth = img2patch(enc_oth, code_block_size, code_block_size, code_block_size)
            paddings = (6, 6, 6, 6)
            enc_oth = F.pad(enc_oth, paddings, "constant")

            paddings = (0, padding_sub_w[j], 0, padding_sub_h[j])
            LL_context = F.pad(LL, paddings, "reflect")
            HL_context = F.pad(HL_list[j], paddings, "reflect")
            paddings = (6, 6, 6, 6)
            LL_context = F.pad(LL_context, paddings, "reflect")
            LL_context = img2patch_padding(LL_context, code_block_size + 12, code_block_size + 12, code_block_size, 6)
            HL_context = F.pad(HL_context, paddings, "reflect")
            HL_context = img2patch_padding(HL_context, code_block_size + 12, code_block_size + 12, code_block_size, 6)

            for h_i in range(code_block_size):
                for w_i in range(code_block_size):
                    cur_ct_y = copy.deepcopy(enc_oth[:, 0:1, h_i:h_i + 13, w_i:w_i + 13])
                    cur_ct_y[:, :, 13 // 2 + 1:13, :] = 0.
                    cur_ct_y[:, :, 13 // 2, 13 // 2:13] = 0.
                    cur_context = LL_context[:, 0:1, h_i:h_i + 13, w_i:w_i + 13]
                    prob_y = models_dict['entropy_HL_Y'][j](cur_ct_y, cur_context,
                                                              yuv_low_bound[3 * j + 1], yuv_high_bound[3 * j + 1])

                    prob_y = prob_y.cpu().data.numpy()
                    prob_y = (prob_y * freqs_resolution).astype(np.int64)
                    index = enc_oth[:, 0, h_i + 6, w_i + 6].cpu().data.numpy().astype(np.int64)

                    for sample_idx, prob_sample in enumerate(prob_y):
                        coe_id = (sample_idx // tmp_hor_num) * tmp_hor_num * code_block_size*code_block_size + \
                                 h_i * tmp_stride + \
                                 (sample_idx % tmp_hor_num) * code_block_size + \
                                 w_i
                        if (coe_id % tmp_stride) < subband_w[j] and (coe_id // tmp_stride) < subband_h[j]:
                            # 不在padding的区域内:
                            if shift_min[0, 3 * j + 1] < shift_max[0, 3 * j + 1]:
                                freqs = ac.SimpleFrequencyTable(
                                    prob_sample[shift_min[0, 3 * j + 1]:shift_max[0, 3 * j + 1] + 1])
                                data = index[sample_idx] - min_v[0, 3 * j + 1]
                                assert data >= 0
                                enc.write(freqs, data)
                            coded_coe_num = coded_coe_num + 1
                                         
            for h_i in range(code_block_size):
                for w_i in range(code_block_size):
                    cur_ct_u = copy.deepcopy(enc_oth[:, 1:2, h_i:h_i + 13, w_i:w_i + 13])
                    cur_context = torch.cat((LL_context[:, 1:2, h_i:h_i + 13, w_i:w_i + 13],
                                             HL_context[:, 0:1, h_i:h_i + 13, w_i:w_i + 13]), dim=1)
                    prob_u = models_dict['entropy_HL_U'][j](cur_ct_u, cur_context,
                                                       yuv_low_bound[3 * j + 1], yuv_high_bound[3 * j + 1])

                    prob_u = prob_u.cpu().data.numpy()
                    prob_u = (prob_u * freqs_resolution).astype(np.int64)
                    index = enc_oth[:, 1, h_i + 6, w_i + 6].cpu().data.numpy().astype(np.int64)

                    for sample_idx, prob_sample in enumerate(prob_u):
                        coe_id = (sample_idx // tmp_hor_num) * tmp_hor_num * code_block_size * code_block_size + \
                                 h_i * tmp_stride + \
                                 (sample_idx % tmp_hor_num) * code_block_size + \
                                 w_i
                        if (coe_id % tmp_stride) < subband_w[j] and (coe_id // tmp_stride) < subband_h[j]:
                            # 不在padding的区域内:
                            if shift_min[1, 3 * j + 1] < shift_max[1, 3 * j + 1]:
                                freqs = ac.SimpleFrequencyTable(
                                    prob_sample[shift_min[1, 3 * j + 1]:shift_max[1, 3 * j + 1] + 1])
                                data = index[sample_idx] - min_v[1, 3 * j + 1]
                                assert data >= 0
                                enc.write(freqs, data)
                            coded_coe_num = coded_coe_num + 1
                            
            for h_i in range(code_block_size):
                for w_i in range(code_block_size):
                    cur_ct_v = copy.deepcopy(enc_oth[:, 2:3, h_i:h_i + 13, w_i:w_i + 13])
                    cur_ct_v[:, :, 13 // 2 + 1:13, :] = 0.
                    cur_ct_v[:, :, 13 // 2, 13 // 2:13] = 0.
                    cur_context = torch.cat((LL_context[:, 2:3, h_i:h_i + 13, w_i:w_i + 13],
                                             HL_context[:, 0:2, h_i:h_i + 13, w_i:w_i + 13]), dim=1)
                    prob_v = models_dict['entropy_HL_V'][j](cur_ct_v, cur_context,
                                                       yuv_low_bound[3 * j + 1], yuv_high_bound[3 * j + 1])

                    prob_v = prob_v.cpu().data.numpy()
                    prob_v = (prob_v * freqs_resolution).astype(np.int64)
                    index = enc_oth[:, 2, h_i + 6, w_i + 6].cpu().data.numpy().astype(np.int64)

                    for sample_idx, prob_sample in enumerate(prob_v):
                        coe_id = (sample_idx // tmp_hor_num) * tmp_hor_num * code_block_size * code_block_size + \
                                 h_i * tmp_stride + \
                                 (sample_idx % tmp_hor_num) * code_block_size + \
                                 w_i
                        if (coe_id % tmp_stride) < subband_w[j] and (coe_id // tmp_stride) < subband_h[j]:
                            # 不在padding的区域内:
                            if shift_min[2, 3 * j + 1] < shift_max[2, 3 * j + 1]:
                                freqs = ac.SimpleFrequencyTable(
                                    prob_sample[shift_min[2, 3 * j + 1]:shift_max[2, 3 * j + 1] + 1])
                                data = index[sample_idx] - min_v[2, 3 * j + 1]
                                assert data >= 0
                                enc.write(freqs, data)
                            coded_coe_num = coded_coe_num + 1

            print('HL' + str(j) + ' encoded...')
            
            HL_list[j] = HL_list[j]*used_scale
            # compress LH
            paddings = (0, padding_sub_w[j], 0, padding_sub_h[j])
            enc_oth = F.pad(LH_list[j], paddings, "constant")
            enc_oth = img2patch(enc_oth, code_block_size, code_block_size, code_block_size)
            paddings = (6, 6, 6, 6)
            enc_oth = F.pad(enc_oth, paddings, "constant")

            paddings = (0, padding_sub_w[j], 0, padding_sub_h[j])
            LL_context = F.pad(LL, paddings, "reflect")
            HL_context = F.pad(HL_list[j], paddings, "reflect")
            LH_context = F.pad(LH_list[j], paddings, "reflect")
            paddings = (6, 6, 6, 6)
            LL_context = F.pad(LL_context, paddings, "reflect")
            LL_context = img2patch_padding(LL_context, code_block_size + 12, code_block_size + 12, code_block_size, 6)
            HL_context = F.pad(HL_context, paddings, "reflect")
            HL_context = img2patch_padding(HL_context, code_block_size + 12, code_block_size + 12, code_block_size, 6)
            LH_context = F.pad(LH_context, paddings, "reflect")
            LH_context = img2patch_padding(LH_context, code_block_size + 12, code_block_size + 12, code_block_size, 6)

            for h_i in range(code_block_size):
                for w_i in range(code_block_size):
                    cur_ct_y = copy.deepcopy(enc_oth[:, 0:1, h_i:h_i + 13, w_i:w_i + 13])
                    cur_ct_y[:, :, 13 // 2 + 1:13, :] = 0.
                    cur_ct_y[:, :, 13 // 2, 13 // 2:13] = 0.
                    cur_context = torch.cat((LL_context[:, 0:1, h_i:h_i + 13, w_i:w_i + 13],
                                             HL_context[:, 0:1, h_i:h_i + 13, w_i:w_i + 13]), dim=1)
                    prob_y = models_dict['entropy_LH_Y'][j](cur_ct_y, cur_context,
                                                         yuv_low_bound[3 * j + 2], yuv_high_bound[3 * j + 2])

                    prob_y = prob_y.cpu().data.numpy()
                    prob_y = (prob_y * freqs_resolution).astype(np.int64)
                    index = enc_oth[:, 0, h_i + 6, w_i + 6].cpu().data.numpy().astype(np.int64)

                    for sample_idx, prob_sample in enumerate(prob_y):
                        coe_id = (sample_idx // tmp_hor_num) * tmp_hor_num * code_block_size * code_block_size + \
                                 h_i * tmp_stride + \
                                 (sample_idx % tmp_hor_num) * code_block_size + \
                                 w_i
                        if (coe_id % tmp_stride) < subband_w[j] and (coe_id // tmp_stride) < subband_h[j]:
                            # 不在padding的区域内:
                            if shift_min[0, 3 * j + 2] < shift_max[0, 3 * j + 2]:
                                freqs = ac.SimpleFrequencyTable(
                                    prob_sample[shift_min[0, 3 * j + 2]:shift_max[0, 3 * j + 2] + 1])
                                data = index[sample_idx] - min_v[0, 3 * j + 2]
                                assert data >= 0
                                enc.write(freqs, data)
                            coded_coe_num = coded_coe_num + 1
                            
            for h_i in range(code_block_size):
                for w_i in range(code_block_size):
                    cur_ct_u = copy.deepcopy(enc_oth[:, 1:2, h_i:h_i + 13, w_i:w_i + 13])
                    cur_ct_u[:, :, 13 // 2 + 1:13, :] = 0.
                    cur_ct_u[:, :, 13 // 2, 13 // 2:13] = 0.
                    cur_context = torch.cat((LL_context[:, 1:2, h_i:h_i + 13, w_i:w_i + 13],
                                             HL_context[:, 1:2, h_i:h_i + 13, w_i:w_i + 13],
                                             LH_context[:, 0:1, h_i:h_i + 13, w_i:w_i + 13]), dim=1)
                    prob_u = models_dict['entropy_LH_U'][j](cur_ct_u, cur_context,
                                                         yuv_low_bound[3 * j + 2], yuv_high_bound[3 * j + 2])

                    prob_u = prob_u.cpu().data.numpy()
                    prob_u = (prob_u * freqs_resolution).astype(np.int64)
                    index = enc_oth[:, 1, h_i + 6, w_i + 6].cpu().data.numpy().astype(np.int64)

                    for sample_idx, prob_sample in enumerate(prob_u):
                        coe_id = (sample_idx // tmp_hor_num) * tmp_hor_num * code_block_size * code_block_size + \
                                 h_i * tmp_stride + \
                                 (sample_idx % tmp_hor_num) * code_block_size + \
                                 w_i
                        if (coe_id % tmp_stride) < subband_w[j] and (coe_id // tmp_stride) < subband_h[j]:
                            # 不在padding的区域内:
                            if shift_min[1, 3 * j + 2] < shift_max[1, 3 * j + 2]:
                                freqs = ac.SimpleFrequencyTable(
                                    prob_sample[shift_min[1, 3 * j + 2]:shift_max[1, 3 * j + 2] + 1])
                                data = index[sample_idx] - min_v[1, 3 * j + 2]
                                assert data >= 0
                                enc.write(freqs, data)
                            coded_coe_num = coded_coe_num + 1
                            
            for h_i in range(code_block_size):
                for w_i in range(code_block_size):
                    cur_ct_v = copy.deepcopy(enc_oth[:, 2:3, h_i:h_i + 13, w_i:w_i + 13])
                    cur_ct_v[:, :, 13 // 2 + 1:13, :] = 0.
                    cur_ct_v[:, :, 13 // 2, 13 // 2:13] = 0.
                    cur_context = torch.cat((LL_context[:, 2:3, h_i:h_i + 13, w_i:w_i + 13],
                                            HL_context[:, 2:3, h_i:h_i + 13, w_i:w_i + 13],
                                            LH_context[:, 0:2, h_i:h_i + 13, w_i:w_i + 13]), dim=1)
                    prob_v = models_dict['entropy_LH_V'][j](cur_ct_v, cur_context,
                                                         yuv_low_bound[3 * j + 2], yuv_high_bound[3 * j + 2])

                    prob_v = prob_v.cpu().data.numpy()
                    prob_v = (prob_v * freqs_resolution).astype(np.int64)
                    index = enc_oth[:, 2, h_i + 6, w_i + 6].cpu().data.numpy().astype(np.int64)

                    for sample_idx, prob_sample in enumerate(prob_v):
                        coe_id = (sample_idx // tmp_hor_num) * tmp_hor_num * code_block_size * code_block_size + \
                                 h_i * tmp_stride + \
                                 (sample_idx % tmp_hor_num) * code_block_size + \
                                 w_i
                        if (coe_id % tmp_stride) < subband_w[j] and (coe_id // tmp_stride) < subband_h[j]:
                            # 不在padding的区域内:
                            if shift_min[2, 3 * j + 2] < shift_max[2, 3 * j + 2]:
                                freqs = ac.SimpleFrequencyTable(
                                    prob_sample[shift_min[2, 3 * j + 2]:shift_max[2, 3 * j + 2] + 1])
                                data = index[sample_idx] - min_v[2, 3 * j + 2]
                                assert data >= 0
                                enc.write(freqs, data)
                            coded_coe_num = coded_coe_num + 1
            print('LH' + str(j) + ' encoded...')
            LH_list[j] = LH_list[j] * used_scale

            # compress HH
            paddings = (0, padding_sub_w[j], 0, padding_sub_h[j])
            enc_oth = F.pad(HH_list[j], paddings, "constant")
            enc_oth = img2patch(enc_oth, code_block_size, code_block_size, code_block_size)
            paddings = (6, 6, 6, 6)
            enc_oth = F.pad(enc_oth, paddings, "constant")

            paddings = (0, padding_sub_w[j], 0, padding_sub_h[j])
            LL_context = F.pad(LL, paddings, "reflect")
            HL_context = F.pad(HL_list[j], paddings, "reflect")
            LH_context = F.pad(LH_list[j], paddings, "reflect")
            HH_context = F.pad(HH_list[j], paddings, "reflect")
            paddings = (6, 6, 6, 6)
            LL_context = F.pad(LL_context, paddings, "reflect")
            LL_context = img2patch_padding(LL_context, code_block_size + 12, code_block_size + 12, code_block_size, 6)
            HL_context = F.pad(HL_context, paddings, "reflect")
            HL_context = img2patch_padding(HL_context, code_block_size + 12, code_block_size + 12, code_block_size, 6)
            LH_context = F.pad(LH_context, paddings, "reflect")
            LH_context = img2patch_padding(LH_context, code_block_size + 12, code_block_size + 12, code_block_size, 6)
            HH_context = F.pad(HH_context, paddings, "reflect")
            HH_context = img2patch_padding(HH_context, code_block_size + 12, code_block_size + 12, code_block_size, 6)

            for h_i in range(code_block_size):
                for w_i in range(code_block_size):
                    cur_ct_y = copy.deepcopy(enc_oth[:, 0:1, h_i:h_i + 13, w_i:w_i + 13])
                    cur_ct_y[:, :, 13 // 2 + 1:13, :] = 0.
                    cur_ct_y[:, :, 13 // 2, 13 // 2:13] = 0.
                    cur_context = torch.cat((LL_context[:, 0:1, h_i:h_i + 13, w_i:w_i + 13],
                                             HL_context[:, 0:1, h_i:h_i + 13, w_i:w_i + 13],
                                             LH_context[:, 0:1, h_i:h_i + 13, w_i:w_i + 13]), dim=1)
                    prob_y = models_dict['entropy_HH_Y'][j](cur_ct_y, cur_context,
                                                         yuv_low_bound[3 * j + 3], yuv_high_bound[3 * j + 3])

                    prob_y = prob_y.cpu().data.numpy()
                    prob_y = (prob_y * freqs_resolution).astype(np.int64)
                    index = enc_oth[:, 0, h_i + 6, w_i + 6].cpu().data.numpy().astype(np.int64)

                    for sample_idx, prob_sample in enumerate(prob_y):
                        coe_id = (sample_idx // tmp_hor_num) * tmp_hor_num * code_block_size * code_block_size + \
                                 h_i * tmp_stride + \
                                 (sample_idx % tmp_hor_num) * code_block_size + \
                                 w_i
                        if (coe_id % tmp_stride) < subband_w[j] and (coe_id // tmp_stride) < subband_h[j]:
                            # 不在padding的区域内:
                            if shift_min[0, 3 * j + 3] < shift_max[0, 3 * j + 3]:
                                freqs = ac.SimpleFrequencyTable(
                                    prob_sample[shift_min[0, 3 * j + 3]:shift_max[0, 3 * j + 3] + 1])
                                data = index[sample_idx] - min_v[0, 3 * j + 3]
                                assert data >= 0
                                enc.write(freqs, data)
                            coded_coe_num = coded_coe_num + 1
                            
            for h_i in range(code_block_size):
                for w_i in range(code_block_size):
                    cur_ct_u = copy.deepcopy(enc_oth[:, 1:2, h_i:h_i + 13, w_i:w_i + 13])
                    cur_ct_u[:, :, 13 // 2 + 1:13, :] = 0.
                    cur_ct_u[:, :, 13 // 2, 13 // 2:13] = 0.
                    cur_context = torch.cat((LL_context[:, 1:2, h_i:h_i + 13, w_i:w_i + 13],
                                             HL_context[:, 1:2, h_i:h_i + 13, w_i:w_i + 13],
                                             LH_context[:, 1:2, h_i:h_i + 13, w_i:w_i + 13],
                                             HH_context[:, 0:1, h_i:h_i + 13, w_i:w_i + 13]), dim=1)
                    prob_u = models_dict['entropy_HH_U'][j](cur_ct_u, cur_context,
                                                         yuv_low_bound[3 * j + 3], yuv_high_bound[3 * j + 3])

                    prob_u = prob_u.cpu().data.numpy()
                    prob_u = (prob_u * freqs_resolution).astype(np.int64)
                    index = enc_oth[:, 1, h_i + 6, w_i + 6].cpu().data.numpy().astype(np.int64)

                    for sample_idx, prob_sample in enumerate(prob_u):
                        coe_id = (sample_idx // tmp_hor_num) * tmp_hor_num * code_block_size * code_block_size + \
                                 h_i * tmp_stride + \
                                 (sample_idx % tmp_hor_num) * code_block_size + \
                                 w_i
                        if (coe_id % tmp_stride) < subband_w[j] and (coe_id // tmp_stride) < subband_h[j]:
                            # 不在padding的区域内:
                            if shift_min[1, 3 * j + 3] < shift_max[1, 3 * j + 3]:
                                freqs = ac.SimpleFrequencyTable(
                                    prob_sample[shift_min[1, 3 * j + 3]:shift_max[1, 3 * j + 3] + 1])
                                data = index[sample_idx] - min_v[1, 3 * j + 3]
                                assert data >= 0
                                enc.write(freqs, data)
                            coded_coe_num = coded_coe_num + 1
                            
            for h_i in range(code_block_size):
                for w_i in range(code_block_size):
                    cur_ct_v = copy.deepcopy(enc_oth[:, 2:3, h_i:h_i + 13, w_i:w_i + 13])
                    cur_ct_v[:, :, 13 // 2 + 1:13, :] = 0.
                    cur_ct_v[:, :, 13 // 2, 13 // 2:13] = 0.
                    cur_context = torch.cat((LL_context[:, 2:3, h_i:h_i + 13, w_i:w_i + 13],
                                             HL_context[:, 2:3, h_i:h_i + 13, w_i:w_i + 13],
                                             LH_context[:, 2:3, h_i:h_i + 13, w_i:w_i + 13],
                                             HH_context[:, 0:2, h_i:h_i + 13, w_i:w_i + 13]), dim=1)
                    prob_v = models_dict['entropy_HH_V'][j](cur_ct_v, cur_context,
                                                         yuv_low_bound[3 * j + 3], yuv_high_bound[3 * j + 3])

                    prob_v = prob_v.cpu().data.numpy()
                    prob_v = (prob_v * freqs_resolution).astype(np.int64)
                    index = enc_oth[:, 2, h_i + 6, w_i + 6].cpu().data.numpy().astype(np.int64)

                    for sample_idx, prob_sample in enumerate(prob_v):
                        coe_id = (sample_idx // tmp_hor_num) * tmp_hor_num * code_block_size * code_block_size + \
                                 h_i * tmp_stride + \
                                 (sample_idx % tmp_hor_num) * code_block_size + \
                                 w_i
                        if (coe_id % tmp_stride) < subband_w[j] and (coe_id // tmp_stride) < subband_h[j]:
                            # 不在padding的区域内:
                            if shift_min[2, 3 * j + 3] < shift_max[2, 3 * j + 3]:
                                freqs = ac.SimpleFrequencyTable(
                                    prob_sample[shift_min[2, 3 * j + 3]:shift_max[2, 3 * j + 3] + 1])
                                data = index[sample_idx] - min_v[2, 3 * j + 3]
                                assert data >= 0
                                enc.write(freqs, data)
                            coded_coe_num = coded_coe_num + 1

            print('HH' + str(j) + ' encoded...')
            HH_list[j] = HH_list[j] * used_scale

            LL = models_dict['transform'][j].inverse_trans(LL, HL_list[j], LH_list[j], HH_list[j])

        assert (coded_coe_num == (height + pad_h) * (width + pad_w) * 3)

        recon = LL
        recon = yuv2rgb(recon)
        recon = recon[:, :, 0:height, 0:width]
        
        h_list = [0, height//3, height//3*2, height]
        w_list = [0, width//3, width//3*2, width]
        k_ = 3
        rgb_post = torch.zeros_like(recon)
        for _i in range(k_):
            for _j in range(k_):
                pad_start_h = max(h_list[_i] - 64, 0) - h_list[_i]
                pad_end_h = min(h_list[_i + 1] + 64, height) - h_list[_i + 1]
                pad_start_w = max(w_list[_j] - 64, 0) - w_list[_j]
                pad_end_w = min(w_list[_j + 1] + 64, width) - w_list[_j + 1]
                tmp = models_dict['post'](recon[:, :, h_list[_i] + pad_start_h:h_list[_i + 1] + pad_end_h,
                                    w_list[_j] + pad_start_w:w_list[_j + 1] + pad_end_w])
                rgb_post[:, :, h_list[_i]:h_list[_i + 1], w_list[_j]:w_list[_j + 1]] = tmp[:, :,
                                                                                            -pad_start_h:tmp.size()[
                                                                                                            2] - pad_end_h,
                                                                                            -pad_start_w:tmp.size()[
                                                                                                            3] - pad_end_w]
        recon = rgb_post

        recon = torch.clamp(torch.round(recon), 0., 255.)
        mse = torch.mean((recon.cpu() - original_img) ** 2)
        psnr = (10. * torch.log10(255. * 255. / mse)).item()

        recon = recon[0, :, :, :]
        recon = recon.permute(1, 2, 0)
        recon = recon.cpu().data.numpy().astype(np.uint8)
        img = Image.fromarray(recon, 'RGB')
        img.save(args.recon_dir + '/' + bin_name + '.png')

        enc.finish()
        
    return height, width, psnr


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='parse hyper parameter'
    )
    parser.add_argument('--model_dir', type=str, default=r'./models')
    parser.add_argument('--log_dir', type=str, default=r'./output')
    parser.add_argument('--bin_dir', type=str, default=r'./output/bin')
    parser.add_argument('--recon_dir', type=str, default=r'./output/recon')

    parser.add_argument('--isLossless', type=int, default=0)
    parser.add_argument('--scale_list', type=float, nargs="+", default=[
        8.0])
    parser.add_argument('--lm', type=float, default = 0.16, help='R-D trade off')
    parser.add_argument('--wavelet_affine', type=bool, default=False,
                        help="the type of wavelet: True:affine False:additive")
    parser.add_argument('--code_block_size', type=int, default=16)
    parser.add_argument('--input_dir', type=str, default=r'./picture')
    parser.add_argument('--img_name', type=str, default=r'kodim01.png')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    enc_lossy(args)