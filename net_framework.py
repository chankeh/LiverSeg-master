import math
import torch
from torch import nn, cat, add
import numpy as np
import torch.nn.functional as F

# UNet3D

class EncoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncoderConv, self).__init__()

        self.conv1 = nn.Conv3d(in_ch, out_ch // 2, 3, padding=1)
        self.conv2 = nn.Conv3d(out_ch // 2, out_ch, 3, padding=1)

        self.bn1 = nn.GroupNorm(4, out_ch // 2)
        self.bn2 = nn.GroupNorm(4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class DecoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DecoderConv, self).__init__()

        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)

        self.bn = nn.GroupNorm(4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet3D, self).__init__()
        self.encoderconv1 = EncoderConv(n_channels, 64)

        self.encoderconv2 = EncoderConv(64, 128)

        self.encoderconv3 = EncoderConv(128, 256)

        self.encoderconv4 = EncoderConv(256, 512)
        self.up1 = nn.ConvTranspose3d(512, 512, 2, stride=2, padding=0)
        self.decoderconv1 = DecoderConv(768, 256)
        self.up2 = nn.ConvTranspose3d(256, 256, 2, stride=2, padding=0)
        self.decoderconv2 = DecoderConv(384, 128)
        self.up3 = nn.ConvTranspose3d(128, 128, 2, stride=2, padding=0)
        self.decoderconv3 = DecoderConv(192, 64)

        self.out_conv = nn.Conv3d(64, n_classes, 1)
        self.maxpooling = nn.MaxPool3d(2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_input):
        x_en1 = self.encoderconv1(x_input)
        x = self.maxpooling(x_en1)
        x_en2 = self.encoderconv2(x)
        x = self.maxpooling(x_en2)
        x_en3 = self.encoderconv3(x)
        x = self.maxpooling(x_en3)
        x_en4 = self.encoderconv4(x)
        x = self.up1(x_en4)
        x_de1 = self.decoderconv1(cat([x_en3, x], dim=1))
        x = self.up2(x_de1)
        x_de2 = self.decoderconv2(cat([x_en2, x], dim=1))
        x = self.up3(x_de2)
        x_de3 = self.decoderconv3(cat([x_en1, x], dim=1))

        x = self.out_conv(x_de3)

        x = self.softmax(x)
        return x


# Attention UNet3D

class UnetConv3(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(3, 3, 3), padding_size=(1, 1, 1), init_stride=(1, 1, 1)):
        super(UnetConv3, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                   nn.GroupNorm(4, out_size),
                                   nn.ReLU(inplace=True), )
        self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                   nn.GroupNorm(4, out_size),
                                   nn.ReLU(inplace=True), )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UnetGridGatingSignal(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(1, 1, 1)):
        super(UnetGridGatingSignal, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv3d(in_ch, out_ch, kernel_size, (1, 1, 1), (0, 0, 0)),
                                   nn.GroupNorm(4, out_ch),
                                   nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


class GridAttentionBlock3D(nn.Module):
    def __init__(self, x_ch, g_ch, sub_sample_factor=(2, 2, 2)):
        super(GridAttentionBlock3D, self).__init__()

        self.W = nn.Sequential(
            nn.Conv3d(x_ch,
                      x_ch,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.GroupNorm(4, x_ch))
        self.theta = nn.Conv3d(x_ch,
                               x_ch,
                               kernel_size=sub_sample_factor,
                               stride=sub_sample_factor,
                               padding=0,
                               bias=False)
        self.phi = nn.Conv3d(g_ch,
                             x_ch,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             bias=True)
        self.psi = nn.Conv3d(x_ch,
                             out_channels=1,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             bias=True)

    def forward(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        phi_g = F.upsample(self.phi(g),
                           size=theta_x_size[2:],
                           mode='trilinear')

        f = F.relu(theta_x + phi_g, inplace=True)

        sigm_psi_f = F.sigmoid(self.psi(f))
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode='trilinear')
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)
        return W_y


class UnetUp3(nn.Module):
    def __init__(self, in_size, out_size):
        super(UnetUp3, self).__init__()

        self.conv = UnetConv3(in_size, out_size)
        self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class AttentionUNet3D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(AttentionUNet3D, self).__init__()

        self.encoderconv1 = UnetConv3(n_channels, 32)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.encoderconv2 = UnetConv3(32, 64)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.encoderconv3 = UnetConv3(64, 128)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.encoderconv4 = UnetConv3(128, 256)
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(256, 512)
        self.gating = UnetGridGatingSignal(512, 256, kernel_size=(1, 1, 1))

        self.attentionblock4 = GridAttentionBlock3D(256, 256)
        self.attentionblock3 = GridAttentionBlock3D(128, 256)
        self.attentionblock2 = GridAttentionBlock3D(64, 256)

        self.up_concat4 = UnetUp3(512, 256)
        self.up_concat3 = UnetUp3(256, 128)
        self.up_concat2 = UnetUp3(128, 64)
        self.up_concat1 = UnetUp3(64, 32)

        self.out_conv = nn.Conv3d(32, n_classes, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_input):
        x_en1 = self.encoderconv1(x_input)
        pool1 = self.maxpool1(x_en1)

        x_en2 = self.encoderconv2(pool1)
        pool2 = self.maxpool2(x_en2)

        x_en3 = self.encoderconv3(pool2)
        pool3 = self.maxpool3(x_en3)

        x_en4 = self.encoderconv4(pool3)
        pool4 = self.maxpool4(x_en4)

        center = self.center(pool4)
        gating = self.gating(center)

        att4 = self.attentionblock4(x_en4, gating)
        att3 = self.attentionblock3(x_en3, gating)
        att2 = self.attentionblock2(x_en2, gating)

        up4 = self.up_concat4(att4, center)
        up3 = self.up_concat3(att3, up4)
        up2 = self.up_concat2(att2, up3)
        up1 = self.up_concat1(x_en1, up2)

        x = self.out_conv(up1)

        x = self.softmax(x)
        return x


# Attention UNet2D

class add_attn(nn.Module):
    def __init__(self, x_channels, g_channels=128):
        super(add_attn, self).__init__()
        self.W = nn.Sequential(
            nn.Conv2d(x_channels,
                      x_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(x_channels))
        self.theta = nn.Conv2d(x_channels,
                               x_channels,
                               kernel_size=2,
                               stride=2,
                               padding=0,
                               bias=False)
        self.phi = nn.Conv2d(g_channels,
                             x_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             bias=True)
        self.psi = nn.Conv2d(x_channels,
                             out_channels=1,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             bias=True)

    def forward(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        phi_g = F.upsample(self.phi(g),
                           size=theta_x_size[2:],
                           mode='bilinear')

        f = F.relu(theta_x + phi_g, inplace=True)

        sigm_psi_f = F.sigmoid(self.psi(f))
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode='bilinear')
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)
        return W_y


class unetCat(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(unetCat, self).__init__()
        self.convT = nn.ConvTranspose2d(dim_in, dim_out, kernel_size=4, stride=2, padding=1)

    def forward(self, input_1, input_2):
        output_2 = self.convT(input_2)
        offset = output_2.size()[2] - input_1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        output_1 = F.pad(input_1, padding)
        y = torch.cat([output_1, output_2], 1)
        return y


class AttentionUNet2D(nn.Module):
    def __init__(self, n_channels, n_classes, useBN=True, pretrained=False): # channel:1  classes:3
        super(AttentionUNet2D, self).__init__()
        self.n_classes = n_classes
        self.conv1 = self.add_conv_stage(n_channels, 1, useBN=useBN)
        self.conv2 = self.add_conv_stage(1, 17, useBN=useBN)
        self.conv3 = self.add_conv_stage(17, 34, useBN=useBN)
        self.conv4 = self.add_conv_stage(34, 68, useBN=useBN)

        self.center = self.add_conv_stage(68, 136, useBN=useBN)
        self.gating = self.add_conv(136, 68, useBN=useBN)

        self.attn_1 = add_attn(x_channels=68)
        self.attn_2 = add_attn(x_channels=34)
        self.attn_3 = add_attn(x_channels=17)

        self.cat_1 = unetCat(dim_in=136, dim_out=68)
        self.cat_2 = unetCat(dim_in=68, dim_out=34)
        self.cat_3 = unetCat(dim_in=34, dim_out=17)
        self.cat_4 = unetCat(dim_in=17, dim_out=1)

        self.conv4m = self.add_conv_stage(136, 68, useBN=useBN)
        self.conv3m = self.add_conv_stage(68, 34, useBN=useBN)
        self.conv2m = self.add_conv_stage(34, 17, useBN=useBN)
        self.conv1m = self.add_conv_stage(17, 1, useBN=useBN)
        self.final_conv = nn.Sequential(
            nn.Conv2d(1, n_classes, 1, 1, 1)

        )  # n_classes 3
        self.max_pool = nn.MaxPool2d(2)# 2-->1
        self.max_pool1 = nn.MaxPool2d(1)


        self.upsample43 = self.upsample(136, 68)
        self.upsample32 = self.upsample(68, 34)
        self.upsample21 = self.upsample(34, 17)
        self.softmax = nn.Softmax(dim=1)

    def add_conv_stage(self,
                       dim_in,
                       dim_out,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       useBN=False):
        if useBN:
            return nn.Sequential(
                nn.Conv2d(
                    dim_in,
                    dim_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False),
                nn.BatchNorm2d(dim_out),
                nn.ReLU(inplace=False),
                nn.Conv2d(
                    dim_out,
                    dim_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False),
                nn.BatchNorm2d(dim_out),
                nn.ReLU(inplace=False))
        else:
            return nn.Sequential(
                nn.Conv2d(
                    dim_in,
                    dim_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False),
                nn.ReLU(inplace=False),
                nn.Conv2d(
                    dim_out,
                    dim_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False), nn.ReLU(inplace=False))

    def upsample(self, ch_coarse, ch_fine, useBN=False):
        if useBN:
            return nn.Sequential(
                nn.ConvTranspose2d(ch_coarse, ch_fine, 2, 2, 0, bias=False),
                nn.BatchNorm2d(ch_fine),
                nn.ReLU(inplace=False)
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(ch_coarse, ch_fine, 2, 2, 0, bias=False),
                nn.ReLU(inplace=False)
            )

    def add_conv(self,
                 dim_in,
                 dim_out,
                 kernel_size=1,
                 stride=1,
                 padding=1,
                 useBN=False):
        if useBN:
            return nn.Sequential(
                nn.Conv2d(
                    dim_in,
                    dim_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False),
                nn.BatchNorm2d(dim_out),
                nn.ReLU(inplace=False)
            )

    def forward(self, x):
        # from IPython import embed;embed()
        conv1_out = self.conv1(x)  # (304,304,16) (512,512,16)
        conv2_out = self.conv2(self.max_pool(conv1_out))  # (152,152,32)  (256,256,32)
        conv3_out = self.conv3(self.max_pool(conv2_out))  # (76,76,64)  (128,128,32)
        conv4_out = self.conv4(self.max_pool(conv3_out))  # (38,38,128)  (64,64,64)

        center_out = self.center(self.max_pool(conv4_out))  # (19,19,256)  (32,32,256)
        gating_out = self.gating(center_out)  # (19,19,128)   (32,32,128)

        attn_1_out = self.attn_1(conv4_out, gating_out)  # (38,38,128)  (64,64,128)
        attn_2_out = self.attn_2(conv3_out, gating_out)  # (76,76,64)   (128,128,64)
        attn_3_out = self.attn_3(conv2_out, gating_out)  # (152,152,32)  (256,256,32)

        cat_1_out = self.cat_1(attn_1_out, center_out)  # (38,38,256)  (64,64,128)
        conv4m_out = self.conv4m(cat_1_out)  # (38,38,128)   (64,64,64)
        cat_2_out = self.cat_2(attn_2_out, conv4m_out)  # (76,76,128)  (128,128,128)
        conv3m_out = self.conv3m(cat_2_out)  # (76,76,64)  (128,128,64)
        cat_3_out = self.cat_3(attn_3_out, conv3m_out)  # (152,152,64)  (256,256,64)
        conv2m_out = self.conv2m(cat_3_out)  # (152,152,32)  (128,128,32)
        cat_4_out = self.cat_4(conv1_out, conv2m_out)  # (304,304,32)  (256,256,32)
        conv1m_out = self.conv1m(cat_4_out)  # (304,304,16)  (256,256,16)

        conv0_out = self.final_conv(conv1m_out)
        out = self.softmax(conv0_out)

        return out

# by  gechen
# class Att_unet(nn.Module):
#     def __init__(self,img_ch=1,output_ch=1):
#         super(Att_unet.self).__init__()
#
#         self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
#         self.Conv2 = conv_block(ch_in=64, ch_out=128)
#         self.Conv3 = conv_block(ch_in=128, ch_out=256)
#         self.Conv4 = conv_block(ch_in=256, ch_out=512)
#         self.Conv5 = conv_block(ch_in=512, ch_out=1024)
#
#         self.Up5 = up_conv(ch_in=1024, ch_out=512)
#         self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
#         self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)
#
#         self.Up4 = up_conv(ch_in=512, ch_out=256)
#         self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
#         self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
#
#         self.Up3 = up_conv(ch_in=256, ch_out=128)
#         self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
#         self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
#
#         self.Up2 = up_conv(ch_in=128, ch_out=64)
#         self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
#         self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
#
#         self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
#
#
#     def forward(self, x):
#         # encoding path
#         x1 = self.Conv1(x)
#
#         x2 = self.Maxpool(x1)
#         x2 = self.Conv2(x2)
#
#         x3 = self.Maxpool(x2)
#         x3 = self.Conv3(x3)
#
#         x4 = self.Maxpool(x3)
#         x4 = self.Conv4(x4)
#
#         x5 = self.Maxpool(x4)
#         x5 = self.Conv5(x5)
#
#         # decoding + concat path
#         d5 = self.Up5(x5)
#         x4 = self.Att5(g=d5, x=x4)
#         d5 = torch.cat((x4, d5), dim=1)
#         d5 = self.Up_conv5(d5)
#
#         d4 = self.Up4(d5)
#         x3 = self.Att4(g=d4, x=x3)
#         d4 = torch.cat((x3, d4), dim=1)
#         d4 = self.Up_conv4(d4)
#
#         d3 = self.Up3(d4)
#         x2 = self.Att3(g=d3, x=x2)
#         d3 = torch.cat((x2, d3), dim=1)
#         d3 = self.Up_conv3(d3)
#
#         d2 = self.Up2(d3)
#         x1 = self.Att2(g=d2, x=x1)
#         d2 = torch.cat((x1, d2), dim=1)
#         d2 = self.Up_conv2(d2)
#
#         d1 = self.Conv_1x1(d2)
#
#         return d1


# class Attention_block(nn.Module):
#     def __init__(self, F_g, F_l, F_int):
#         super(Attention_block, self).__init__()
#         self.W_g = nn.Sequential(
#             nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(F_int)
#         )
#
#         self.W_x = nn.Sequential(
#             nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(F_int)
#         )
#
#         self.psi = nn.Sequential(
#             nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )
#
#         self.relu = nn.ReLU(inplace=True)

# test
#
# from torchsummary import summary
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
# model = AttentionUNet3D(1, 3)
# model = model.to(device)
# summary(model, (1, 16, 144, 144))
