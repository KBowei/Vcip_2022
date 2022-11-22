import torch
from torch.autograd import Variable
import glob as gb
from PIL import Image
import numpy as np
from torch.nn import functional as F
import pdb
import arithmetic_coding as ac
import os
from utils import img2patch, img2patch_padding, rgb2yuv, yuv2rgb, find_min_and_max, patch2img, qp_shifts, model_lambdas
from Model import Model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


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


def dec_lossy(args, bin_name, dec, freqs_resolution, logfile):

    trans_steps = 4
    code_block_size = args.code_block_size

    with torch.no_grad():

        freqs = ac.SimpleFrequencyTable(np.ones([8], dtype=np.int))
        model_qp = dec.read(freqs)

        freqs = ac.SimpleFrequencyTable(np.ones([64], dtype=np.int))
        qp_shift = dec.read(freqs)

        init_scale = qp_shifts[model_qp][qp_shift]
        print(init_scale)
        logfile.write(str(init_scale) + '\n')
        logfile.flush()

        # checkpoint = torch.load(args.model_path + '/' + str(model_lambdas[model_qp]) + '_40.pth')
        checkpoint = torch.load('/model/chenhang/Iwave_model/affine_0.0625/model_epoch040_hard.pth')

        all_part_dict = checkpoint['state_dict']
        
        models_dict = {}

        model = Model(wavelet_affine=True).to(device)
        model.load_state_dict(all_part_dict)

        models_dict['scale_net'] = model.scale_net
        models_dict['transform0'] = model.wavelet_transform[0]
        models_dict['transform1'] = model.wavelet_transform[1]
        models_dict['transform2'] = model.wavelet_transform[2]
        models_dict['transform3'] = model.wavelet_transform[3]

        models_dict['transform'] = [models_dict['transform0'], models_dict['transform1'], models_dict['transform2'],
                                    models_dict['transform3']]
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

        height = dec_binary(dec, 15)
        width = dec_binary(dec, 15)

        pad_h = int(np.ceil(height / 16)) * 16 - height
        pad_w = int(np.ceil(width / 16)) * 16 - width

        LL = torch.zeros([1, 3, (height + pad_h) // 16, (width + pad_w) // 16], dtype=torch.float32).cuda()
        HL_list = []
        LH_list = []
        HH_list = []
        down_scales = [2, 4, 8, 16]
        for i in range(trans_steps):
            HL_list.append(torch.zeros([1, 3, (height + pad_h) // down_scales[i],
                                       (width + pad_w) // down_scales[i]], dtype=torch.float32).cuda())
            LH_list.append(torch.zeros([1, 3, (height + pad_h) // down_scales[i],
                                       (width + pad_w) // down_scales[i]], dtype=torch.float32).cuda())
            HH_list.append(torch.zeros([1, 3, (height + pad_h) // down_scales[i],
                                       (width + pad_w) // down_scales[i]], dtype=torch.float32).cuda())
        min_v = np.zeros(shape=(3, 13), dtype=np.int64)
        max_v = np.zeros(shape=(3, 13), dtype=np.int64)
        for i in range(3):
            for j in range(13):
                min_v[i, j] = dec_binary(dec, 15) - 8000
                max_v[i, j] = dec_binary(dec, 15) - 8000
        yuv_low_bound = min_v.min(axis=0)  # 每个子带YUV三个通道最小值的最小值
        yuv_high_bound = max_v.max(axis=0)
        shift_min = min_v - yuv_low_bound  # 对各个子带YUV三个通道的范围减去子带内最小的最小值来偏移 结果：每个子带内部YUV的最小值为0
        shift_max = max_v - yuv_low_bound

        subband_h = [(height + pad_h) // 2, (height + pad_h) // 4, (height + pad_h) // 8, (height + pad_h) // 16]
        subband_w = [(width + pad_w) // 2, (width + pad_w) // 4, (width + pad_w) // 8, (width + pad_w) // 16]
        padding_sub_h = [(int(np.ceil(tmp / code_block_size)) * code_block_size - tmp) for tmp in subband_h]
        padding_sub_w = [(int(np.ceil(tmp / code_block_size)) * code_block_size - tmp) for tmp in subband_w]

        used_scale = init_scale + models_dict['scale_net']()

        coded_coe_num = 0

        # decompress LL
        tmp_stride = subband_w[3] + padding_sub_w[3]
        tmp_hor_num = tmp_stride // code_block_size
        paddings = (0, padding_sub_w[3], 0, padding_sub_h[3])
        enc_LL = F.pad(LL, paddings, "constant")  # 为了裁块做的padding
        enc_LL = img2patch(enc_LL, code_block_size, code_block_size, code_block_size)
        paddings = (6, 6, 6, 6)  # 上下文模型的padding
        enc_LL = F.pad(enc_LL, paddings, "constant")

        for h_i in range(code_block_size):
            for w_i in range(code_block_size):
                cur_ct = enc_LL[:, 0:1, h_i:h_i + 13, w_i:w_i + 13]
                prob_y = models_dict['entropy_LL_Y'](cur_ct, yuv_low_bound[0], yuv_high_bound[0])
                prob_y = prob_y.cpu().data.numpy()

                index = []

                prob_y = (prob_y * freqs_resolution).astype(np.int64)
                for sample_idx, prob_sample in enumerate(prob_y):
                    coe_id = (sample_idx // tmp_hor_num) * tmp_hor_num * code_block_size * code_block_size + \
                             h_i * tmp_stride + \
                             (sample_idx % tmp_hor_num) * code_block_size + \
                             w_i  # 找到当前的解码点在子带内的索引
                    if (coe_id % tmp_stride) < subband_w[3] and (coe_id // tmp_stride) < subband_h[3]:  # 子带内而非padding部分
                        if shift_min[0, 0] < shift_max[0, 0]:
                            # 频率范围裁剪，例如YUV总的range是[10, 215], Y的range_y是[10, 100],
                            # 那么只需取range中索引idx = [0, 90]的元素即可，重点就是索引从0开始
                            freqs = ac.SimpleFrequencyTable(
                                prob_sample[shift_min[0, 0]:shift_max[0, 0] + 1])
                            dec_c = dec.read(freqs) + min_v[0, 0]
                        else:
                            dec_c = min_v[0, 0]
                        coded_coe_num = coded_coe_num + 1
                        index.append(dec_c)
                    else:
                        index.append(0)  
                enc_LL[:, 0, h_i + 6, w_i + 6] = torch.from_numpy(np.array(index).astype(np.float)).cuda()
            
        LL = enc_LL[:, :, 6:-6, 6:-6]
        LL = patch2img(LL, subband_h[3] + padding_sub_h[3], subband_w[3] + padding_sub_w[3])
        LL = LL[:, :, 0:subband_h[3], 0:subband_w[3]]
        print('LL_Y decoded')

        paddings = (0, padding_sub_w[3], 0, padding_sub_h[3])
        context = F.pad(LL, paddings, "reflect")
        paddings = (6, 6, 6, 6)
        context = F.pad(context, paddings, "reflect")

        LL_U_context = img2patch_padding(context[:, 0:1, :, :], code_block_size+12, code_block_size+12, code_block_size, 6)
        for h_i in range(code_block_size):
            for w_i in range(code_block_size):
                cur_ct = enc_LL[:, 1:2, h_i:h_i + 13, w_i:w_i + 13]
                cur_context = LL_U_context[:, :, h_i:h_i + 13, w_i:w_i + 13]
                prob_u = models_dict['entropy_LL_U'](cur_ct, cur_context, yuv_low_bound[0], yuv_high_bound[0])
                prob_u = prob_u.cpu().data.numpy()
                index = []
                prob_u = (prob_u * freqs_resolution).astype(np.int64)
                for sample_idx, prob_sample in enumerate(prob_u):
                    coe_id = (sample_idx // tmp_hor_num) * tmp_hor_num * code_block_size * code_block_size + \
                             h_i * tmp_stride + \
                             (sample_idx % tmp_hor_num) * code_block_size + \
                             w_i
                    if (coe_id % tmp_stride) < subband_w[3] and (coe_id // tmp_stride) < subband_h[3]:
                        if shift_min[1, 0] < shift_max[1, 0]:
                            freqs = ac.SimpleFrequencyTable(
                                prob_sample[shift_min[1, 0]:shift_max[1, 0] + 1])
                            dec_c = dec.read(freqs) + min_v[1, 0]
                        else:
                            dec_c = min_v[1, 0]
                        coded_coe_num = coded_coe_num + 1
                        index.append(dec_c)
                    else:
                        index.append(0)
                enc_LL[:, 1, h_i + 6, w_i + 6] = torch.from_numpy(np.array(index).astype(np.float)).cuda()

        LL = enc_LL[:, :, 6:-6, 6:-6]
        LL = patch2img(LL, subband_h[3] + padding_sub_h[3], subband_w[3] + padding_sub_w[3])
        LL = LL[:, :, 0:subband_h[3], 0:subband_w[3]]
        print('LL_U decoded')

        paddings = (0, padding_sub_w[3], 0, padding_sub_h[3])
        context = F.pad(LL, paddings, "reflect")
        paddings = (6, 6, 6, 6)
        context = F.pad(context, paddings, "reflect")

        LL_V_context = img2patch_padding(context[:, 0:2, :, :], code_block_size+12, code_block_size+12, code_block_size, 6)
        for h_i in range(code_block_size):
            for w_i in range(code_block_size):
                cur_ct = enc_LL[:, 2:3, h_i:h_i + 13, w_i:w_i + 13]
                cur_context = LL_V_context[:, :, h_i:h_i + 13, w_i:w_i + 13]
                prob_v = models_dict['entropy_LL_V'](cur_ct, cur_context, yuv_low_bound[0], yuv_high_bound[0])
                prob_v = prob_v.cpu().data.numpy()

                index = []

                prob_v = (prob_v * freqs_resolution).astype(np.int64)
                for sample_idx, prob_sample in enumerate(prob_v):
                    coe_id = (sample_idx // tmp_hor_num) * tmp_hor_num * code_block_size * code_block_size + \
                             h_i * tmp_stride + \
                             (sample_idx % tmp_hor_num) * code_block_size + \
                             w_i
                    if (coe_id % tmp_stride) < subband_w[3] and (coe_id // tmp_stride) < subband_h[3]:
                        if shift_min[2, 0] < shift_max[2, 0]:
                            freqs = ac.SimpleFrequencyTable(
                                prob_sample[shift_min[2, 0]:shift_max[2, 0] + 1])
                            dec_c = dec.read(freqs) + min_v[2, 0]
                        else:
                            dec_c = min_v[2, 0]
                        coded_coe_num = coded_coe_num + 1
                        index.append(dec_c)
                    else:
                        index.append(0)
                enc_LL[:, 2, h_i + 6, w_i + 6] = torch.from_numpy(np.array(index).astype(np.float)).cuda()
        LL = enc_LL[:, :, 6:-6, 6:-6]
        LL = patch2img(LL, subband_h[3] + padding_sub_h[3], subband_w[3] + padding_sub_w[3])
        LL = LL[:, :, 0:subband_h[3], 0:subband_w[3]]
        print('LL_V decoded')
        print('LL decoded')
        
        LL = LL * used_scale

        for i in range(trans_steps):
            j = trans_steps - 1 - i
            tmp_stride = subband_w[j] + padding_sub_w[j]
            tmp_hor_num = tmp_stride // code_block_size
            # compress HL
            paddings = (0, padding_sub_w[j], 0, padding_sub_h[j])
            enc_oth = F.pad(HL_list[j], paddings, "constant")
            enc_oth = img2patch(enc_oth, code_block_size, code_block_size, code_block_size)
            paddings = (6, 6, 6, 6)
            enc_oth = F.pad(enc_oth, paddings, "constant")

            paddings = (0, padding_sub_w[j], 0, padding_sub_h[j])
            context = F.pad(LL, paddings, "reflect")
            paddings = (6, 6, 6, 6)
            LL = F.pad(context, paddings, "reflect")

            HL_Y_context = img2patch_padding(LL[:, 0:1, :, :], code_block_size+12, code_block_size+12, code_block_size, 6)
            for h_i in range(code_block_size):
                for w_i in range(code_block_size):
                    cur_ct = enc_oth[:, 0:1, h_i:h_i + 13, w_i:w_i + 13]
                    cur_ct[:, :, 6, 6] = 0.
                    cur_context = HL_Y_context[:, :, h_i:h_i + 13, w_i:w_i + 13]
                    prob_y = models_dict['entropy_HL_Y'][j](cur_ct, cur_context,
                                                     yuv_low_bound[3 * j + 1], yuv_high_bound[3 * j + 1])
                    prob_y = prob_y.cpu().data.numpy()

                    index = []

                    prob_y = (prob_y * freqs_resolution).astype(np.int64)

                    for sample_idx, prob_sample in enumerate(prob_y):
                        coe_id = (sample_idx // tmp_hor_num) * tmp_hor_num * code_block_size * code_block_size + \
                                 h_i * tmp_stride + \
                                 (sample_idx % tmp_hor_num) * code_block_size + \
                                 w_i
                        if (coe_id % tmp_stride) < subband_w[j] and (coe_id // tmp_stride) < subband_h[j]:
                            if shift_min[0, 3 * j + 1] < shift_max[0, 3 * j + 1]:
                                freqs = ac.SimpleFrequencyTable(
                                    prob_sample[shift_min[0, 3 * j + 1]:shift_max[0, 3 * j + 1] + 1])
                                dec_c = dec.read(freqs) + min_v[0, 3 * j + 1]
                            else:
                                dec_c = min_v[0, 3 * j + 1]
                            coded_coe_num = coded_coe_num + 1
                            index.append(dec_c)
                        else:
                            index.append(0)
                    enc_oth[:, 0, h_i + 6, w_i + 6] = torch.from_numpy(np.array(index).astype(np.float)).cuda()

            HL_list[j] = enc_oth[:, :, 6:-6, 6:-6]
            HL_list[j] = patch2img(HL_list[j], subband_h[j] + padding_sub_h[j], subband_w[j] + padding_sub_w[j])
            HL_list[j] = HL_list[j][:, :, 0:subband_h[j], 0:subband_w[j]]
            print('HL' + str(j) + '_Y decoded')

            paddings = (0, padding_sub_w[j], 0, padding_sub_h[j])
            context = F.pad(HL_list[j], paddings, "reflect")
            paddings = (6, 6, 6, 6)
            HL_list[j] = F.pad(context, paddings, "reflect")

            HL_U_context = img2patch_padding(torch.cat((LL[:, 1:2, :, :], HL_list[j][:, 0:1, :, :]), dim=1),
                                             code_block_size+12, code_block_size+12, code_block_size, 6)
            for h_i in range(code_block_size):
                for w_i in range(code_block_size):
                    cur_ct = enc_oth[:, 1:2, h_i:h_i + 13, w_i:w_i + 13]
                    cur_ct[:, :, 6, 6] = 0.
                    cur_context = HL_U_context[:, :, h_i:h_i + 13, w_i:w_i + 13]
                    prob_u = models_dict['entropy_HL_U'][j](cur_ct, cur_context,
                                                     yuv_low_bound[3 * j + 1], yuv_high_bound[3 * j + 1])
                    prob_u = prob_u.cpu().data.numpy()

                    index = []

                    prob_u = (prob_u * freqs_resolution).astype(np.int64)

                    for sample_idx, prob_sample in enumerate(prob_u):
                        coe_id = (sample_idx // tmp_hor_num) * tmp_hor_num * code_block_size * code_block_size + \
                                 h_i * tmp_stride + \
                                 (sample_idx % tmp_hor_num) * code_block_size + \
                                 w_i
                        if (coe_id % tmp_stride) < subband_w[j] and (coe_id // tmp_stride) < subband_h[j]:
                            if shift_min[1, 3 * j + 1] < shift_max[1, 3 * j + 1]:
                                freqs = ac.SimpleFrequencyTable(
                                    prob_sample[shift_min[1, 3 * j + 1]:shift_max[1, 3 * j + 1] + 1])
                                dec_c = dec.read(freqs) + min_v[1, 3 * j + 1]
                            else:
                                dec_c = min_v[1, 3 * j + 1]
                            coded_coe_num = coded_coe_num + 1
                            index.append(dec_c)
                        else:
                            index.append(0)
                    enc_oth[:, 1, h_i + 6, w_i + 6] = torch.from_numpy(np.array(index).astype(np.float)).cuda()
            
            HL_list[j] = enc_oth[:, :, 6:-6, 6:-6]
            HL_list[j] = patch2img(HL_list[j], subband_h[j] + padding_sub_h[j], subband_w[j] + padding_sub_w[j])
            HL_list[j] = HL_list[j][:, :, 0:subband_h[j], 0:subband_w[j]]
            print('HL' + str(j) + '_U decoded')

            paddings = (0, padding_sub_w[j], 0, padding_sub_h[j])
            context = F.pad(HL_list[j], paddings, "reflect")
            paddings = (6, 6, 6, 6)
            HL_list[j] = F.pad(context, paddings, "reflect")

            HL_V_context = img2patch_padding(torch.cat((LL[:, 2:3, :, :], HL_list[j][:, 0:2, :, :]), dim=1),
                                             code_block_size + 12, code_block_size + 12, code_block_size, 6)
            for h_i in range(code_block_size):
                for w_i in range(code_block_size):
                    cur_ct = enc_oth[:, 2:3, h_i:h_i + 13, w_i:w_i + 13]
                    cur_ct[:, :, 6, 6] = 0.
                    cur_context = HL_V_context[:, :, h_i:h_i + 13, w_i:w_i + 13]
                    prob_v = models_dict['entropy_HL_V'][j](cur_ct, cur_context,
                                                     yuv_low_bound[3 * j + 1], yuv_high_bound[3 * j + 1])
                    prob_v = prob_v.cpu().data.numpy()

                    index = []

                    prob_v = (prob_v * freqs_resolution).astype(np.int64)

                    for sample_idx, prob_sample in enumerate(prob_v):
                        coe_id = (sample_idx // tmp_hor_num) * tmp_hor_num * code_block_size * code_block_size + \
                                 h_i * tmp_stride + \
                                 (sample_idx % tmp_hor_num) * code_block_size + \
                                 w_i
                        if (coe_id % tmp_stride) < subband_w[j] and (coe_id // tmp_stride) < subband_h[j]:
                            if shift_min[2, 3 * j + 1] < shift_max[2, 3 * j + 1]:
                                freqs = ac.SimpleFrequencyTable(
                                    prob_sample[shift_min[2, 3 * j + 1]:shift_max[2, 3 * j + 1] + 1])
                                dec_c = dec.read(freqs) + min_v[2, 3 * j + 1]
                            else:
                                dec_c = min_v[2, 3 * j + 1]
                            coded_coe_num = coded_coe_num + 1
                            index.append(dec_c)
                        else:
                            index.append(0)
                    enc_oth[:, 2, h_i + 6, w_i + 6] = torch.from_numpy(np.array(index).astype(np.float)).cuda()
            HL_list[j] = enc_oth[:, :, 6:-6, 6:-6]
            HL_list[j] = patch2img(HL_list[j], subband_h[j] + padding_sub_h[j], subband_w[j] + padding_sub_w[j])
            HL_list[j] = HL_list[j][:, :, 0:subband_h[j], 0:subband_w[j]]
            print('HL' + str(j) + '_V decoded')
            print('HL' + str(j) + ' decoded')

            HL_list[j] = HL_list[j] * used_scale

            # compress LH
            paddings = (0, padding_sub_w[j], 0, padding_sub_h[j])
            enc_oth = F.pad(LH_list[j], paddings, "constant")
            enc_oth = img2patch(enc_oth, code_block_size, code_block_size, code_block_size)
            paddings = (6, 6, 6, 6)
            enc_oth = F.pad(enc_oth, paddings, "constant")

            paddings = (0, padding_sub_w[j], 0, padding_sub_h[j])
            context = F.pad(HL_list[j], paddings, "reflect")
            paddings = (6, 6, 6, 6)
            context2 = F.pad(context, paddings, "reflect")

            LH_Y_context = img2patch_padding(torch.cat((LL[:, 0:1, :, :], context2[:, 0:1, :, :]), dim=1), code_block_size + 12,
                                             code_block_size + 12, code_block_size, 6)
            for h_i in range(code_block_size):
                for w_i in range(code_block_size):
                    cur_ct = enc_oth[:, 0:1, h_i:h_i + 13, w_i:w_i + 13]
                    cur_ct[:, :, 6, 6] = 0.
                    cur_context = LH_Y_context[:, :, h_i:h_i + 13, w_i:w_i + 13]
                    prob_y = models_dict['entropy_LH_Y'][j](cur_ct, cur_context,
                                                            yuv_low_bound[3 * j + 2], yuv_high_bound[3 * j + 2])
                    prob_y = prob_y.cpu().data.numpy()

                    index = []

                    prob_y = (prob_y * freqs_resolution).astype(np.int64)

                    for sample_idx, prob_sample in enumerate(prob_y):
                        coe_id = (sample_idx // tmp_hor_num) * tmp_hor_num * code_block_size * code_block_size + \
                                 h_i * tmp_stride + \
                                 (sample_idx % tmp_hor_num) * code_block_size + \
                                 w_i
                        if (coe_id % tmp_stride) < subband_w[j] and (coe_id // tmp_stride) < subband_h[j]:
                            if shift_min[0, 3 * j + 2] < shift_max[0, 3 * j + 2]:
                                freqs = ac.SimpleFrequencyTable(
                                    prob_sample[shift_min[0, 3 * j + 2]:shift_max[0, 3 * j + 2] + 1])
                                dec_c = dec.read(freqs) + min_v[0, 3 * j + 2]
                            else:
                                dec_c = min_v[0, 3 * j + 2]
                            coded_coe_num = coded_coe_num + 1
                            index.append(dec_c)
                        else:
                            index.append(0)
                    enc_oth[:, 0, h_i + 6, w_i + 6] = torch.from_numpy(np.array(index).astype(np.float)).cuda()

            LH_list[j] = enc_oth[:, :, 6:-6, 6:-6]
            LH_list[j] = patch2img(LH_list[j], subband_h[j] + padding_sub_h[j], subband_w[j] + padding_sub_w[j])
            LH_list[j] = LH_list[j][:, :, 0:subband_h[j], 0:subband_w[j]]
            print('LH' + str(j) + '_Y decoded')

            paddings = (0, padding_sub_w[j], 0, padding_sub_h[j])
            context = F.pad(HL_list[j], paddings, "reflect")
            paddings = (6, 6, 6, 6)
            context2 = F.pad(context, paddings, "reflect")

            paddings = (0, padding_sub_w[j], 0, padding_sub_h[j])
            context = F.pad(LH_list[j], paddings, "reflect")
            paddings = (6, 6, 6, 6)
            context3 = F.pad(context, paddings, "reflect")

            LH_U_context = img2patch_padding(torch.cat((LL[:, 1:2], context2[:, 1:2], context3[:, 0:1]), dim=1),
                                             code_block_size + 12, code_block_size + 12, code_block_size, 6)
            for h_i in range(code_block_size):
                for w_i in range(code_block_size):
                    cur_ct = enc_oth[:, 1:2, h_i:h_i + 13, w_i:w_i + 13]
                    cur_ct[:, :, 6, 6] = 0.
                    cur_context = LH_U_context[:, :, h_i:h_i + 13, w_i:w_i + 13]
                    prob_u = models_dict['entropy_LH_U'][j](cur_ct, cur_context,
                                                            yuv_low_bound[3 * j + 2], yuv_high_bound[3 * j + 2])
                    prob_u = prob_u.cpu().data.numpy()

                    index = []

                    prob_u = (prob_u * freqs_resolution).astype(np.int64)

                    for sample_idx, prob_sample in enumerate(prob_u):
                        coe_id = (sample_idx // tmp_hor_num) * tmp_hor_num * code_block_size * code_block_size + \
                                 h_i * tmp_stride + \
                                 (sample_idx % tmp_hor_num) * code_block_size + \
                                 w_i
                        if (coe_id % tmp_stride) < subband_w[j] and (coe_id // tmp_stride) < subband_h[j]:
                            if shift_min[1, 3 * j + 2] < shift_max[1, 3 * j + 2]:
                                freqs = ac.SimpleFrequencyTable(
                                    prob_sample[shift_min[1, 3 * j + 2]:shift_max[1, 3 * j + 2] + 1])
                                dec_c = dec.read(freqs) + min_v[1, 3 * j + 2]
                            else:
                                dec_c = min_v[1, 3 * j + 2]
                            coded_coe_num = coded_coe_num + 1
                            index.append(dec_c)
                        else:
                            index.append(0)
                    enc_oth[:, 1, h_i + 6, w_i + 6] = torch.from_numpy(np.array(index).astype(np.float)).cuda()
            LH_list[j] = enc_oth[:, :, 6:-6, 6:-6]
            LH_list[j] = patch2img(LH_list[j], subband_h[j] + padding_sub_h[j], subband_w[j] + padding_sub_w[j])
            LH_list[j] = LH_list[j][:, :, 0:subband_h[j], 0:subband_w[j]]
            print('LH' + str(j) + '_U decoded')

            paddings = (0, padding_sub_w[j], 0, padding_sub_h[j])
            context = F.pad(HL_list[j], paddings, "reflect")
            paddings = (6, 6, 6, 6)
            context2 = F.pad(context, paddings, "reflect")

            paddings = (0, padding_sub_w[j], 0, padding_sub_h[j])
            context = F.pad(LH_list[j], paddings, "reflect")
            paddings = (6, 6, 6, 6)
            context3 = F.pad(context, paddings, "reflect")

            LH_V_context = img2patch_padding(torch.cat((LL[:, 2:3], context2[:, 2:3], context3[:, 0:2]), dim=1),
                                             code_block_size + 12, code_block_size + 12, code_block_size, 6)
            for h_i in range(code_block_size):
                for w_i in range(code_block_size):
                    cur_ct = enc_oth[:, 2:3, h_i:h_i + 13, w_i:w_i + 13]
                    cur_ct[:, :, 6, 6] = 0.
                    cur_context = LH_V_context[:, :, h_i:h_i + 13, w_i:w_i + 13]
                    prob_v = models_dict['entropy_LH_V'][j](cur_ct, cur_context,
                                                            yuv_low_bound[3 * j + 2], yuv_high_bound[3 * j + 2])
                    prob_v = prob_v.cpu().data.numpy()

                    index = []

                    prob_v = (prob_v * freqs_resolution).astype(np.int64)

                    for sample_idx, prob_sample in enumerate(prob_v):
                        coe_id = (sample_idx // tmp_hor_num) * tmp_hor_num * code_block_size * code_block_size + \
                                 h_i * tmp_stride + \
                                 (sample_idx % tmp_hor_num) * code_block_size + \
                                 w_i
                        if (coe_id % tmp_stride) < subband_w[j] and (coe_id // tmp_stride) < subband_h[j]:
                            if shift_min[2, 3 * j + 2] < shift_max[2, 3 * j + 2]:
                                freqs = ac.SimpleFrequencyTable(
                                    prob_sample[shift_min[2, 3 * j + 2]:shift_max[2, 3 * j + 2] + 1])
                                dec_c = dec.read(freqs) + min_v[2, 3 * j + 2]
                            else:
                                dec_c = min_v[2, 3 * j + 2]
                            coded_coe_num = coded_coe_num + 1
                            index.append(dec_c)
                        else:
                            index.append(0)
                    enc_oth[:, 2, h_i + 6, w_i + 6] = torch.from_numpy(np.array(index).astype(np.float)).cuda()

            LH_list[j] = enc_oth[:, :, 6:-6, 6:-6]
            LH_list[j] = patch2img(LH_list[j], subband_h[j] + padding_sub_h[j], subband_w[j] + padding_sub_w[j])
            LH_list[j] = LH_list[j][:, :, 0:subband_h[j], 0:subband_w[j]]

            print('LH' + str(j) + '_V decoded')
            print('LH' + str(j) + ' decoded')

            LH_list[j] = LH_list[j] * used_scale

            # compress HH
            paddings = (0, padding_sub_w[j], 0, padding_sub_h[j])
            enc_oth = F.pad(HH_list[j], paddings, "constant")
            enc_oth = img2patch(enc_oth, code_block_size, code_block_size, code_block_size)
            paddings = (6, 6, 6, 6)
            enc_oth = F.pad(enc_oth, paddings, "constant")

            paddings = (0, padding_sub_w[j], 0, padding_sub_h[j])
            context = F.pad(HL_list[j], paddings, "reflect")
            paddings = (6, 6, 6, 6)
            context2 = F.pad(context, paddings, "reflect")

            paddings = (0, padding_sub_w[j], 0, padding_sub_h[j])
            context = F.pad(LH_list[j], paddings, "reflect")
            paddings = (6, 6, 6, 6)
            context3 = F.pad(context, paddings, "reflect")

            HH_Y_context = img2patch_padding(torch.cat((LL[:, 0:1], context2[:, 0:1], context3[:, 0:1]), dim=1), code_block_size + 12,
                                             code_block_size + 12, code_block_size, 6)
            for h_i in range(code_block_size):
                for w_i in range(code_block_size):
                    cur_ct = enc_oth[:, 0:1, h_i:h_i + 13, w_i:w_i + 13]
                    cur_ct[:, :, 6, 6] = 0.
                    cur_context = HH_Y_context[:, :, h_i:h_i + 13, w_i:w_i + 13]
                    prob_y = models_dict['entropy_HH_Y'][j](cur_ct, cur_context,
                                                            yuv_low_bound[3 * j + 3], yuv_high_bound[3 * j + 3])
                    prob_y = prob_y.cpu().data.numpy()

                    index = []

                    prob_y = (prob_y * freqs_resolution).astype(np.int64)

                    for sample_idx, prob_sample in enumerate(prob_y):
                        coe_id = (sample_idx // tmp_hor_num) * tmp_hor_num * code_block_size * code_block_size + \
                                 h_i * tmp_stride + \
                                 (sample_idx % tmp_hor_num) * code_block_size + \
                                 w_i
                        if (coe_id % tmp_stride) < subband_w[j] and (coe_id // tmp_stride) < subband_h[j]:
                            if shift_min[0, 3 * j + 3] < shift_max[0, 3 * j + 3]:
                                freqs = ac.SimpleFrequencyTable(
                                    prob_sample[shift_min[0, 3 * j + 3]:shift_max[0, 3 * j + 3] + 1])
                                dec_c = dec.read(freqs) + min_v[0, 3 * j + 3]
                            else:
                                dec_c = min_v[0, 3 * j + 3]
                            coded_coe_num = coded_coe_num + 1
                            index.append(dec_c)
                        else:
                            index.append(0)
                    enc_oth[:, 0, h_i + 6, w_i + 6] = torch.from_numpy(np.array(index).astype(np.float)).cuda()
                    
            HH_list[j] = enc_oth[:, :, 6:-6, 6:-6]
            HH_list[j] = patch2img(HH_list[j], subband_h[j] + padding_sub_h[j], subband_w[j] + padding_sub_w[j])
            HH_list[j] = HH_list[j][:, :, 0:subband_h[j], 0:subband_w[j]]
            print('HH' + str(j) + '_Y decoded')

            paddings = (0, padding_sub_w[j], 0, padding_sub_h[j])
            context = F.pad(HL_list[j], paddings, "reflect")
            paddings = (6, 6, 6, 6)
            context2 = F.pad(context, paddings, "reflect")

            paddings = (0, padding_sub_w[j], 0, padding_sub_h[j])
            context = F.pad(LH_list[j], paddings, "reflect")
            paddings = (6, 6, 6, 6)
            context3 = F.pad(context, paddings, "reflect")

            paddings = (0, padding_sub_w[j], 0, padding_sub_h[j])
            context = F.pad(HH_list[j], paddings, "reflect")
            paddings = (6, 6, 6, 6)
            context4 = F.pad(context, paddings, "reflect")

            HH_U_context = img2patch_padding(torch.cat((LL[:, 1:2], context2[:, 1:2], context3[:, 1:2], context4[:, 0:1]), dim=1),
                                             code_block_size + 12, code_block_size + 12, code_block_size, 6)
            for h_i in range(code_block_size):
                for w_i in range(code_block_size):
                    cur_ct = enc_oth[:, 1:2, h_i:h_i + 13, w_i:w_i + 13]
                    cur_ct[:, :, 6, 6] = 0.
                    cur_context = HH_U_context[:, :, h_i:h_i + 13, w_i:w_i + 13]
                    prob_u = models_dict['entropy_HH_U'][j](cur_ct, cur_context,
                                                            yuv_low_bound[3 * j + 3], yuv_high_bound[3 * j + 3])
                    prob_u = prob_u.cpu().data.numpy()

                    index = []

                    prob_u = (prob_u * freqs_resolution).astype(np.int64)

                    for sample_idx, prob_sample in enumerate(prob_u):
                        coe_id = (sample_idx // tmp_hor_num) * tmp_hor_num * code_block_size * code_block_size + \
                                 h_i * tmp_stride + \
                                 (sample_idx % tmp_hor_num) * code_block_size + \
                                 w_i
                        if (coe_id % tmp_stride) < subband_w[j] and (coe_id // tmp_stride) < subband_h[j]:
                            if shift_min[1, 3 * j + 3] < shift_max[1, 3 * j + 3]:
                                freqs = ac.SimpleFrequencyTable(
                                    prob_sample[shift_min[1, 3 * j + 3]:shift_max[1, 3 * j + 3] + 1])
                                dec_c = dec.read(freqs) + min_v[1, 3 * j + 3]
                            else:
                                dec_c = min_v[1, 3 * j + 3]
                            coded_coe_num = coded_coe_num + 1
                            index.append(dec_c)
                        else:
                            index.append(0)
                    enc_oth[:, 1, h_i + 6, w_i + 6] = torch.from_numpy(np.array(index).astype(np.float)).cuda()
                    
            HH_list[j] = enc_oth[:, :, 6:-6, 6:-6]
            HH_list[j] = patch2img(HH_list[j], subband_h[j] + padding_sub_h[j], subband_w[j] + padding_sub_w[j])
            HH_list[j] = HH_list[j][:, :, 0:subband_h[j], 0:subband_w[j]]
            print('HH' + str(j) + '_U decoded')

            paddings = (0, padding_sub_w[j], 0, padding_sub_h[j])
            context = F.pad(HL_list[j], paddings, "reflect")
            paddings = (6, 6, 6, 6)
            context2 = F.pad(context, paddings, "reflect")

            paddings = (0, padding_sub_w[j], 0, padding_sub_h[j])
            context = F.pad(LH_list[j], paddings, "reflect")
            paddings = (6, 6, 6, 6)
            context3 = F.pad(context, paddings, "reflect")

            paddings = (0, padding_sub_w[j], 0, padding_sub_h[j])
            context = F.pad(HH_list[j], paddings, "reflect")
            paddings = (6, 6, 6, 6)
            context4 = F.pad(context, paddings, "reflect")

            HH_V_context = img2patch_padding(torch.cat((LL[:, 2:3], context2[:, 2:3], context3[:, 2:3], context4[:, 0:2]), dim=1),
                                             code_block_size + 12, code_block_size + 12, code_block_size, 6)
            for h_i in range(code_block_size):
                for w_i in range(code_block_size):
                    cur_ct = enc_oth[:, 2:3, h_i:h_i + 13, w_i:w_i + 13]
                    cur_ct[:, :, 6, 6] = 0.
                    cur_context = HH_V_context[:, :, h_i:h_i + 13, w_i:w_i + 13]
                    prob_v = models_dict['entropy_HH_V'][j](cur_ct, cur_context,
                                                            yuv_low_bound[3 * j + 3], yuv_high_bound[3 * j + 3])
                    prob_v = prob_v.cpu().data.numpy()

                    index = []

                    prob_v = (prob_v * freqs_resolution).astype(np.int64)

                    for sample_idx, prob_sample in enumerate(prob_v):
                        coe_id = (sample_idx // tmp_hor_num) * tmp_hor_num * code_block_size * code_block_size + \
                                 h_i * tmp_stride + \
                                 (sample_idx % tmp_hor_num) * code_block_size + \
                                 w_i
                        if (coe_id % tmp_stride) < subband_w[j] and (coe_id // tmp_stride) < subband_h[j]:
                            if shift_min[2, 3 * j + 3] < shift_max[2, 3 * j + 3]:
                                freqs = ac.SimpleFrequencyTable(
                                    prob_sample[shift_min[2, 3 * j + 3]:shift_max[2, 3 * j + 3] + 1])
                                dec_c = dec.read(freqs) + min_v[2, 3 * j + 3]
                            else:
                                dec_c = min_v[2, 3 * j + 3]
                            coded_coe_num = coded_coe_num + 1
                            index.append(dec_c)
                        else:
                            index.append(0)
                    enc_oth[:, 2, h_i + 6, w_i + 6] = torch.from_numpy(np.array(index).astype(np.float)).cuda()

            HH_list[j] = enc_oth[:, :, 6:-6, 6:-6]
            HH_list[j] = patch2img(HH_list[j], subband_h[j] + padding_sub_h[j], subband_w[j] + padding_sub_w[j])
            HH_list[j] = HH_list[j][:, :, 0:subband_h[j], 0:subband_w[j]]

            print('HH' + str(j) + '_V decoded')
            print('HH' + str(j) + ' decoded')

            HH_list[j] = HH_list[j] * used_scale

            LL = LL[:, :, 6:-6, 6:-6]
            LL = LL[:, :, 0:subband_h[j], 0:subband_w[j]]

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

        recon = recon[0, :, :, :]
        recon = recon.permute(1, 2, 0)
        recon = recon.cpu().data.numpy().astype(np.uint8)
        img = Image.fromarray(recon, 'RGB')
        img.save(args.recon_dir + '/' + bin_name + 'affine.png')

    logfile.flush()
