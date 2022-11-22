# 直接
import torch
from torch.nn import functional as F

lifting_coeff = [-1.586134342059924, -0.052980118572961, 0.882911075530934, 0.443506852043971, 0.869864451624781,
                 1.149604398860241]  # bior4.4


class P_block(torch.nn.Module):
    def __init__(self):
        super(P_block, self).__init__()
        self.padding_reflect = torch.nn.ReflectionPad2d(1)
        self.conv1 = torch.nn.Conv2d(3, 16, 3, 1, 0)  # 没有初始化
        self.conv2 = torch.nn.Conv2d(16, 16, 3, 1, 0)
        self.conv3 = torch.nn.Conv2d(16, 16, 3, 1, 0)
        self.conv4 = torch.nn.Conv2d(16, 3, 3, 1, 0)
        self.act = torch.nn.Tanh()

    def forward(self, x):
        x1 = self.conv1(self.padding_reflect(x))
        x2 = self.conv2(self.padding_reflect(self.act(x1)))
        x3 = self.conv3(self.padding_reflect(self.act(x2)))
        x4 = x1 + x3
        x5 = self.conv4(self.padding_reflect(x4))
        return x5


class learn_lifting97(torch.nn.Module):
    def __init__(self, trainable_set):
        super(learn_lifting97, self).__init__()
        if trainable_set:
            self.leran_wavelet_rate = 0.1
        else:
            self.leran_wavelet_rate = 0.0
        self.skip1 = torch.nn.Conv2d(3, 3, (3, 1), padding=0, bias=False, groups=3)
        self.skip1.weight = torch.nn.Parameter(torch.Tensor([[[[0.0], [lifting_coeff[0]], [lifting_coeff[0]]]],
                                                             [[[0.0], [lifting_coeff[0]], [lifting_coeff[0]]]],
                                                             [[[0.0], [lifting_coeff[0]], [lifting_coeff[0]]]]]),
                                               requires_grad=trainable_set)
        self.p_block1 = P_block()

        self.skip2 = torch.nn.Conv2d(3, 3, (3, 1), padding=0, bias=False, groups=3)
        self.skip2.weight = torch.nn.Parameter(torch.Tensor([[[[lifting_coeff[1]], [lifting_coeff[1]], [0.0]]],
                                                             [[[lifting_coeff[1]], [lifting_coeff[1]], [0.0]]],
                                                             [[[lifting_coeff[1]], [lifting_coeff[1]], [0.0]]]]),
                                               requires_grad=trainable_set)
        self.p_block2 = P_block()

        self.skip3 = torch.nn.Conv2d(3, 3, (3, 1), padding=0, bias=False, groups=3)
        self.skip3.weight = torch.nn.Parameter(torch.Tensor([[[[0.0], [lifting_coeff[2]], [lifting_coeff[2]]]],
                                                             [[[0.0], [lifting_coeff[2]], [lifting_coeff[2]]]],
                                                             [[[0.0], [lifting_coeff[2]], [lifting_coeff[2]]]]]),
                                               requires_grad=trainable_set)
        self.p_block3 = P_block()

        self.skip4 = torch.nn.Conv2d(3, 3, (3, 1), padding=0, bias=False, groups=3)
        self.skip4.weight = torch.nn.Parameter(torch.Tensor([[[[lifting_coeff[3]], [lifting_coeff[3]], [0.0]]],
                                                             [[[lifting_coeff[3]], [lifting_coeff[3]], [0.0]]],
                                                             [[[lifting_coeff[3]], [lifting_coeff[3]], [0.0]]]]),
                                               requires_grad=trainable_set)
        self.p_block4 = P_block()

        self.n_h = torch.nn.Parameter(torch.Tensor([0]), requires_grad=trainable_set)
        self.n_l = torch.nn.Parameter(torch.Tensor([0]), requires_grad=trainable_set)

    def forward_trans(self, L, H):

        paddings = (0, 0, 1, 1)

        tmp = F.pad(L, paddings, "reflect")
        skip1 = self.skip1(tmp)
        L_net = self.p_block1(skip1 / 256.) * 256.
        H = H + skip1 + L_net * self.leran_wavelet_rate

        tmp = F.pad(H, paddings, "reflect")
        skip2 = self.skip2(tmp)
        H_net = self.p_block2(skip2 / 256.) * 256.
        L = L + skip2 + H_net * self.leran_wavelet_rate

        tmp = F.pad(L, paddings, "reflect")
        skip3 = self.skip3(tmp)
        L_net = self.p_block3(skip3 / 256.) * 256.
        H = H + skip3 + L_net * self.leran_wavelet_rate

        tmp = F.pad(H, paddings, "reflect")
        skip4 = self.skip4(tmp)
        H_net = self.p_block4(skip4 / 256.) * 256.
        L = L + skip4 + H_net * self.leran_wavelet_rate

        H = H * (lifting_coeff[4] + self.n_h * self.leran_wavelet_rate)
        L = L * (lifting_coeff[5] + self.n_l * self.leran_wavelet_rate)

        return L, H

    def inverse_trans(self, L, H):

        H = H / (lifting_coeff[4] + self.n_h * self.leran_wavelet_rate)
        L = L / (lifting_coeff[5] + self.n_l * self.leran_wavelet_rate)

        paddings = (0, 0, 1, 1)

        tmp = F.pad(H, paddings, "reflect")
        skip4 = self.skip4(tmp)
        H_net = self.p_block4(skip4 / 256.) * 256.
        L = L - skip4 - H_net * self.leran_wavelet_rate

        tmp = F.pad(L, paddings, "reflect")
        skip3 = self.skip3(tmp)
        L_net = self.p_block3(skip3 / 256.) * 256.
        H = H - skip3 - L_net * self.leran_wavelet_rate

        tmp = F.pad(H, paddings, "reflect")
        skip2 = self.skip2(tmp)
        H_net = self.p_block2(skip2 / 256.) * 256.
        L = L - skip2 - H_net * self.leran_wavelet_rate

        tmp = F.pad(L, paddings, "reflect")
        skip1 = self.skip1(tmp)
        L_net = self.p_block1(skip1 / 256.) * 256.
        H = H - skip1 - L_net * self.leran_wavelet_rate

        return L, H


class Wavelet(torch.nn.Module):
    def __init__(self, trainable_set):
        super(Wavelet, self).__init__()

        self.lifting = learn_lifting97(trainable_set)

    def forward_trans(self, x):
        # transform for rows
        L = x[:, :, 0::2, :]
        H = x[:, :, 1::2, :]
        L, H = self.lifting.forward_trans(L, H)

        L = L.permute(0, 1, 3, 2)
        LL = L[:, :, 0::2, :]
        HL = L[:, :, 1::2, :]
        LL, HL = self.lifting.forward_trans(LL, HL)
        LL = LL.permute(0, 1, 3, 2)
        HL = HL.permute(0, 1, 3, 2)

        H = H.permute(0, 1, 3, 2)
        LH = H[:, :, 0::2, :]
        HH = H[:, :, 1::2, :]
        LH, HH = self.lifting.forward_trans(LH, HH)
        LH = LH.permute(0, 1, 3, 2)
        HH = HH.permute(0, 1, 3, 2)

        return LL, HL, LH, HH

    def inverse_trans(self, LL, HL, LH, HH):
        LH = LH.permute(0, 1, 3, 2)
        HH = HH.permute(0, 1, 3, 2)
        H = torch.zeros(LH.size()[0], LH.size()[1], LH.size()[2] + HH.size()[2], LH.size()[3], device=LH.device)
        LH, HH = self.lifting.inverse_trans(LH, HH)
        H[:, :, 0::2, :] = LH
        H[:, :, 1::2, :] = HH
        H = H.permute(0, 1, 3, 2)

        LL = LL.permute(0, 1, 3, 2)
        HL = HL.permute(0, 1, 3, 2)
        L = torch.zeros(LL.size()[0], LL.size()[1], LL.size()[2] + HL.size()[2], LL.size()[3], device=LH.device)
        LL, HL = self.lifting.inverse_trans(LL, HL)
        L[:, :, 0::2, :] = LL
        L[:, :, 1::2, :] = HL
        L = L.permute(0, 1, 3, 2)

        L, H = self.lifting.inverse_trans(L, H)
        x = torch.zeros(L.size()[0], L.size()[1], L.size()[2] + H.size()[2], L.size()[3], device=LH.device)
        x[:, :, 0::2, :] = L
        x[:, :, 1::2, :] = H

        return x
