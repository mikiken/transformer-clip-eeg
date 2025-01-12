import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from functools import partial
import logging
import math
import typing as tp

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

import typing as tp
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=8,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()

        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(2440, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out

class SqueezeLayer(nn.Module):
    def __init__(self, axis):
        super(SqueezeLayer, self).__init__()
        self.axis = axis

    def forward(self, x):
        return torch.squeeze(x, dim=self.axis)

class MelModel(nn.Module):
    def __init__(self, spatial_filters=8, filters_cnn=16, kerSize_temporal=9, stride_temporal=3,
                 units_lstm=32, padding='valid', dropout_rate=0, activation=nn.ReLU(), speech_dim=28):
        super(MelModel, self).__init__()
        self.padding = padding

        self.batchnorm_mel1 = nn.BatchNorm1d(speech_dim)
        self.speech_conv1d = nn.Sequential(nn.Conv1d(speech_dim, spatial_filters, kernel_size=1), activation)
        self.batchnorm_mel2 = nn.BatchNorm1d(spatial_filters)
        self.speech_conv2d = nn.Sequential(nn.Conv2d(1, filters_cnn, kernel_size=(kerSize_temporal, 1),
                                      stride=(stride_temporal, 1), padding=padding), activation)

        self.speech_lstm = nn.LSTM(spatial_filters * filters_cnn, units_lstm, batch_first=True)
        self.spatial_filters = spatial_filters
        self.filters_cnn = filters_cnn

        self.output_dim = units_lstm
        self.stride_temporal = stride_temporal
        self.kernel_size = kerSize_temporal


    def get_output_dim(self, input_window_size):
        if self.padding == 'valid':
            return int((input_window_size - self.kernel_size) / self.stride_temporal + 1) * self.output_dim
        else:
            return int(input_window_size / self.stride_temporal) * self.output_dim


    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = self.batchnorm_mel1(x)
        x = self.speech_conv1d(x)
        x = self.batchnorm_mel2(x)
        x = torch.unsqueeze(x, 1)
        x = torch.permute(x, (0, 1, 3, 2))
        x = self.speech_conv2d(x)
        x = torch.permute(x, (0, 2, 1, 3))
        x = torch.reshape(x, [x.shape[0], x.shape[1], self.spatial_filters * self.filters_cnn])
        x, _ = self.speech_lstm(x)

        return x

class Wav2vecSmallModel(nn.Module):
    def __init__(self, spatial_filters=64, ks_temporal=3, stride_temporal=3,
                dropout_rate=0, activation=nn.LeakyReLU(), speech_dim=1024):
        super(Wav2vecSmallModel, self).__init__()

        self.batchnorm_mel1 = nn.BatchNorm1d(speech_dim)
        self.speech_conv1d = nn.Sequential(nn.Conv1d(speech_dim, spatial_filters,
                                                     kernel_size=ks_temporal, stride=stride_temporal, padding='valid'), activation)

        self.output_dim = spatial_filters



    def get_output_dim(self, input_window_size):

        return int(input_window_size * self.output_dim)


    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = self.batchnorm_mel1(x)
        x = self.speech_conv1d(x)
        x = torch.permute(x, (0, 2, 1))

        return x


class SpeechSmallConv(nn.Module):
    def __init__(self, output_dim=64, ks_temporal=20,
                dropout_rate=0.2, speech_dim=1024, time_dimension=64*5):
        super(SpeechSmallConv, self).__init__()


        # spatio-temporal rearrangement of the speech features
        self.speech_spatial_mapping = nn.Conv1d(speech_dim, output_dim, kernel_size=ks_temporal, padding='same')
        # add dropout rate
        self.dropout = nn.Dropout(dropout_rate)
        self.layernorm = nn.LayerNorm([output_dim, time_dimension])
        self.activation = nn.LeakyReLU()

        self.output_dim = output_dim

    def get_output_dim(self, input_window_size):

        return int(input_window_size * self.output_dim)


    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = self.speech_spatial_mapping(x)
        x = self.dropout(x)
        x = self.layernorm(x)
        x = self.activation(x)
        x = torch.permute(x, (0, 2, 1))

        return x

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=64, time_dimension=320, dropout_rate=0.2,stride=1, padding= 'same', dilation=1, activation=nn.LeakyReLU()):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalization = nn.LayerNorm([out_channels, time_dimension])
        # self.activation = nn.LeakyReLU() # used for all old experiments, up until 1 may 2024
        self.activation = nn.GELU()


    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x

class EEGConvLSTM(nn.Module):
    def __init__(self,
                 units_lstm=128,
                 output_dim = 64,
                 dropout_rate=0.2, eeg_dim=64,
                 filters=(256, 256, 256, 128, 128),
                 kernels=(64,) * 5,
                 dilation_rate=1,
                 input_channels=64,
                 time_dimension=64 * 5,
                 normalization_fn='layer_norm',
                 activation_fn='leaky_relu',
                 ):
        super(EEGConvLSTM, self).__init__()


        self.speech_lstm1 = nn.LSTM(filters[-1], units_lstm, batch_first=True, bidirectional=True)
        self.speech_lstm2 = nn.LSTM(units_lstm*2, int(output_dim/2), batch_first=True, bidirectional=True)
        self.spatial_filters = input_channels
        self.output_dim = output_dim

        self.eeg_spatial_mapping = nn.Conv1d(eeg_dim, filters[0], kernel_size=1)  # Identity mapping for eeg
        self.n_blocks = len(filters)


        for i in range(len(filters)):
            filter_ = filters[i]
            kernel = kernels[i]
            name = f'conv_{i}'

            convBlock = BasicBlock(in_channels=filter_, out_channels=filter_, kernel_size=kernel, dilation=dilation_rate, time_dimension=time_dimension, dropout_rate=dropout_rate)

            setattr(self, name, convBlock)

            # dilation = dilation_rate
            #
            # conv = nn.Conv1d(in_channels=input_channels, out_channels=filter_, kernel_size=kernel, padding='same',dilation=dilation)
            # # add dropout rate
            # dropout = nn.Dropout(dropout_rate)
            # normalization = nn.LayerNorm([filter_, time_dimension])
            # activation = nn.LeakyReLU()
            #
            # seq = nn.Sequential(conv, dropout, normalization, activation)




    def get_output_dim(self, input_window_size):
        return input_window_size*self.output_dim


    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = self.eeg_spatial_mapping(x)

        eeg_x = x

        eeg_x.to(x.device)
        # spatial remapping

        for i in range(self.n_blocks):
            layer = getattr(self, f'conv_{i}')

            # don't add skip connection in the last block
            if i == self.n_blocks - 1:
                x = layer(x)
            else:
                # add a skip connection after the each layer and before the activation,
                x = layer(x+eeg_x)
        # permute
        x = torch.permute(x, (0, 2, 1))
        x, _ = self.speech_lstm1(x)
        x, _ = self.speech_lstm2(x)

        return x

class EEGConformer(nn.Module):
    def __init__(self,
                 output_dim = 8,
                 conformer_input_dim = 64,
                 dropout_rate=0.2, eeg_dim=64,
                 filters=(64,) * 2,
                 kernels=(64,) * 2,
                 dilation_rate=1,
                 input_channels=64,
                 time_dimension=64 * 5,
                 depth=2,
                 normalization_fn='layer_norm',
                 activation_fn='leaky_relu',
                 ):
        super(EEGConformer, self).__init__()



        self.spatial_filters = input_channels
        self.output_dim = output_dim

        self.eeg_spatial_mapping = nn.Conv1d(eeg_dim, filters[0], kernel_size=1)  # Identity mapping for eeg
        self.n_blocks = len(filters)


        for i in range(len(filters)):
            filter_ = filters[i]
            kernel = kernels[i]
            name = f'conv_{i}'

            convBlock = BasicBlock(in_channels=filter_, out_channels=filter_, kernel_size=kernel, dilation=dilation_rate, time_dimension=time_dimension, dropout_rate=dropout_rate)

            setattr(self, name, convBlock)

        # after the convolution, we add the attention block
        self.transformerEncoder = TransformerEncoder(depth, conformer_input_dim)

        # final layer to get to output dimension
        self.final_layer = nn.Linear(conformer_input_dim, output_dim)



    def get_output_dim(self, input_window_size):
        return input_window_size*self.output_dim


    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = self.eeg_spatial_mapping(x)

        eeg_x = x

        eeg_x.to(x.device)
        # spatial remapping

        for i in range(self.n_blocks):
            layer = getattr(self, f'conv_{i}')

            # don't add skip connection in the last block
            if i == self.n_blocks - 1:
                x = layer(x)
            else:
                # add a skip connection after the each layer and before the activation,
                x = layer(x+eeg_x)
        # permute
        x = torch.permute(x, (0, 2, 1))

        x = self.transformerEncoder(x)

        x = self.final_layer(x)

        return x

class EEGConformerInterleaved(nn.Module):
    def __init__(self,
                 output_dim = 8,
                 conformer_input_dim = 64,
                 dropout_rate=0.2, eeg_dim=64,
                 filters=(64,) *1,
                 kernels=(64,) * 1,
                 dilation_rate=1,
                 input_channels=64,
                 time_dimension=64 * 5,
                 depth=4,
                 normalization_fn='layer_norm',
                 activation_fn='leaky_relu',
                 ):
        super(EEGConformerInterleaved, self).__init__()



        self.spatial_filters = input_channels
        self.output_dim = output_dim

        self.eeg_spatial_mapping = nn.Conv1d(eeg_dim, filters[0], kernel_size=1)  # Identity mapping for eeg
        self.n_blocks = depth


        for i in range(depth):
            filter_ = filters[0]
            kernel = kernels[0]
            name = f'conv_{i}'
            convBlock = BasicBlock(in_channels=filter_, out_channels=filter_, kernel_size=kernel, dilation=dilation_rate, time_dimension=time_dimension, dropout_rate=dropout_rate)
            setattr(self, name, convBlock)

            name = f'conformer_{i}'
            # after the convolution, we add the attention block
            transformerEncoder = TransformerEncoder(1, conformer_input_dim)
            setattr(self, name, transformerEncoder)


        # final layer to get to output dimension
        self.final_layer = nn.Linear(conformer_input_dim, output_dim)

    def get_output_dim(self, input_window_size):
        return input_window_size*self.output_dim


    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = self.eeg_spatial_mapping(x)
        eeg_x = x
        eeg_x.to(x.device)
        eeg_x_conformer = torch.permute(eeg_x, (0, 2, 1))
        eeg_x_conformer.to(x.device)
        # spatial remapping

        for i in range(self.n_blocks):

            layer = getattr(self, f'conv_{i}')
            if i != 0:
                x = torch.permute(x, (0, 2, 1))
            x = layer(x + eeg_x)

            x = torch.permute(x, (0, 2, 1))
            layer_conformer = getattr(self, f'conformer_{i}')

            # don't add skip connection in the last block
            if i == self.n_blocks - 1:
                x = layer_conformer(x)
            else:
                # add a skip connection after the each layer and before the activation,
                x = layer_conformer(x+eeg_x_conformer)


        x = self.final_layer(x)

        return x

class EEGConvLSTMNew(nn.Module):
    def __init__(self,
                 output_dim = 128,
                 dropout_rate=0.2, eeg_dim=64,
                 filters=(256,)*2,
                 kernels=(64,) * 2,
                 dilation_rate=1,
                 input_channels=64,
                 time_dimension=64 * 5,
                 normalization_fn='layer_norm',
                 activation_fn='leaky_relu',
                 ):
        super(EEGConvLSTMNew, self).__init__()



        self.lstm = nn.LSTM(input_channels, int(output_dim/2), batch_first=True, bidirectional=True)
        self.spatial_filters = input_channels
        self.output_dim = output_dim

        self.eeg_spatial_mapping = nn.Conv1d(eeg_dim, eeg_dim, kernel_size=1)  # Identity mapping for eeg
        self.n_blocks = len(filters)


        for i in range(len(filters)):
            filter_ = filters[i]
            kernel = kernels[i]
            name = f'conv_{i}'

            convBlock = BasicBlock(in_channels=input_channels, out_channels=filter_, kernel_size=kernel, dilation=dilation_rate, time_dimension=time_dimension, dropout_rate=dropout_rate)

            setattr(self, name, convBlock)

            # dilation = dilation_rate
            #
            # conv = nn.Conv1d(in_channels=input_channels, out_channels=filter_, kernel_size=kernel, padding='same',dilation=dilation)
            # # add dropout rate
            # dropout = nn.Dropout(dropout_rate)
            # normalization = nn.LayerNorm([filter_, time_dimension])
            # activation = nn.LeakyReLU()
            #
            # seq = nn.Sequential(conv, dropout, normalization, activation)




    def get_output_dim(self, input_window_size):
        return input_window_size*self.output_dim


    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        eeg_x = x

        eeg_x.to(x.device)
        for i in range(self.n_blocks):
            layer = getattr(self, f'conv_{i}')

            # don't add skip connection in the last block
            if i == self.n_blocks - 1:
                x = layer(x)
            else:
                # add a skip connection after the each layer and before the activation,
                x = layer(x+eeg_x)
        # permute
        x = torch.permute(x, (0, 2, 1))
        x, _ = self.lstm(x)


        return x

class EEGModel(nn.Module):
    def __init__(self, spatial_filters_eeg=32, filters_cnn_eeg=16, kerSize_temporal=9, stride_temporal=3,
                 units_hidden=128, units_lstm=32, fun_act=nn.ReLU(), padding='valid'):
        super(EEGModel, self).__init__()

        self.batchnorm = nn.BatchNorm1d(   64 )
        self.eeg_conv1d = nn.Sequential(nn.Conv1d(64, spatial_filters_eeg, kernel_size=1), fun_act)
        self.batchnorm_eeg = nn.BatchNorm1d(spatial_filters_eeg)
        self.eeg_conv2d = nn.Sequential(nn.Conv2d(1, filters_cnn_eeg, kernel_size=(kerSize_temporal, 1),
                                      stride=(stride_temporal, 1), padding=padding), fun_act)
        self.eeg_td1 = nn.Sequential(nn.Linear(spatial_filters_eeg * filters_cnn_eeg, units_hidden), fun_act)
        self.eeg_td2 = nn.Sequential(nn.Linear(units_hidden, units_lstm), fun_act)
        self.spatial_filters_eeg = spatial_filters_eeg
        self.filters_cnn_eeg = filters_cnn_eeg

        self.output_dim = units_lstm
        self.stride_temporal = stride_temporal
        self.kernel_size = kerSize_temporal

    def get_output_dim(self, input_window_size):
        return int((input_window_size- self.kernel_size)/self.stride_temporal+1)*self.output_dim

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = self.batchnorm(x)
        x = self.eeg_conv1d(x)
        x = self.batchnorm_eeg(x)
        x = torch.unsqueeze(x, 1)
        x = torch.permute(x, (0, 1,3, 2))
        x = self.eeg_conv2d(x)
        x = torch.permute(x, (0, 2, 1,3))
        x = torch.reshape(x,[x.shape[0], x.shape[1], self.spatial_filters_eeg * self.filters_cnn_eeg])
        x = self.eeg_td1(x)
        x = self.eeg_td2(x)
        return x

class EEGLstm(nn.Module):
    def __init__(self, spatial_filters=32,
                 units_lstm=64, padding='valid',
                 dropout_rate=0, activation=nn.LeakyReLU(), speech_dim=64):
        super(EEGLstm, self).__init__()

        self.batchnorm_mel1 = nn.BatchNorm1d(speech_dim)
        self.speech_conv1d = nn.Sequential(nn.Conv1d(speech_dim, spatial_filters, kernel_size=1), activation)
        self.batchnorm_mel2 = nn.BatchNorm1d(spatial_filters)

        self.speech_lstm1 = nn.LSTM(spatial_filters, units_lstm, batch_first=True)
        self.speech_lstm2 = nn.LSTM(units_lstm, units_lstm, batch_first=True)
        self.spatial_filters = spatial_filters
        self.output_dim = units_lstm


    def get_output_dim(self, input_window_size):
        return input_window_size*self.output_dim


    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = self.batchnorm_mel1(x)
        x = self.speech_conv1d(x)
        x = self.batchnorm_mel2(x)
        # permute
        x = torch.permute(x, (0, 2, 1))
        x, _ = self.speech_lstm1(x)
        x, _ = self.speech_lstm2(x)

        return x

class EEGExtended(nn.Module):
    def __init__(self, spatial_filters_eeg=32, filters_cnn_eeg=16, kerSize_temporal=9, stride_temporal=3,
                 units_hidden=128, units_lstm=32, fun_act=nn.ReLU(), padding='valid'):
        super(EEGExtended, self).__init__()

        self.batchnorm = nn.BatchNorm1d(64)
        self.eeg_conv1d = nn.Sequential(nn.Conv1d(64, spatial_filters_eeg, kernel_size=1), fun_act)
        self.batchnorm_eeg = nn.BatchNorm1d(spatial_filters_eeg)
        self.eeg_conv2d = nn.Sequential(nn.Conv2d(1, filters_cnn_eeg, kernel_size=(kerSize_temporal, 1),
                                                  stride=(stride_temporal, 1), padding=padding), fun_act)
        self.eeg_td1 = nn.Sequential(nn.Linear(spatial_filters_eeg * filters_cnn_eeg, units_hidden), fun_act)
        self.eeg_td2 = nn.Sequential(nn.Linear(units_hidden, units_lstm), fun_act)
        self.spatial_filters_eeg = spatial_filters_eeg
        self.filters_cnn_eeg = filters_cnn_eeg

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = self.batchnorm(x)
        x = self.eeg_conv1d(x)
        x = self.batchnorm_eeg(x)
        x = torch.unsqueeze(x, 1)
        x = torch.permute(x, (0, 1, 3, 2))
        x = self.eeg_conv2d(x)
        x = torch.permute(x, (0, 2, 1, 3))
        x = torch.reshape(x, [x.shape[0], x.shape[1], self.spatial_filters_eeg * self.filters_cnn_eeg])
        x = self.eeg_td1(x)
        x = self.eeg_td2(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation=nn.ReLU()):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        return x

class CLIP(nn.Module):
    def __init__(self, eegModel, speechModel, temperature=1.):
        super(CLIP, self).__init__()
        self.eegModel = eegModel
        self.speechModel = speechModel
        self.temperature = nn.Parameter(torch.tensor(temperature))


    def forward(self, eeg,speech):
        # Getting Image and Text Features
        EEG_features = self.eegModel(eeg)
        speech_features = self.speechModel(speech)

        # Getting Image and Text Embeddings (with same dimension)
        EEG_features = torch.flatten(EEG_features, start_dim=1)
        speech_features = torch.flatten(speech_features, start_dim=1)
        
        # l2-normalize them
        EEG_features = F.normalize(EEG_features, p=2, dim=1)
        speech_features = F.normalize(speech_features, p=2, dim=1)

        # Calculating the Loss
        logits = (speech_features @ EEG_features.T) * torch.exp(self.temperature)
        # EEGs_similarity = EEG_embeddings @ EEG_embeddings.T
        # speechs_similarity = speech_embeddings @ speech_embeddings.T

        # targets but in torch
        targets = torch.arange(logits.shape[0], device=logits.device)

        #
        # texts_loss = cross_entropy(logits, targets, reduction='none')
        # images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        speech_loss = F.cross_entropy(logits, targets)
        EEG_loss = F.cross_entropy(logits.T, targets.T)
        # loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)
        loss = (speech_loss + EEG_loss) / 2.0
        return loss.mean()

    # PyTor

class memoryBank(nn.Module):
    def __init__(self, bank_size: int ,
                 device: torch.device,
                 dim:int,
                 momentum: float =0.90):
        super(memoryBank, self).__init__()
        self.bank_size = bank_size # memory bank size
        self.dim = dim # latent dim
        self.momentum = momentum# update rate

        self._init_mem_bank(bank_size, dim, device)

        # register a buffer in the form of a dictionary

    def _init_mem_bank(self, bank_size: int, dim: int, device:torch.device) -> None:
        """
        Given the memory bank size and the channel dimension, initialize the memory
            bank.
        Args:
            bank_size (int): size of the memory bank, expected to be the same size as
                 the training set.
            dim (int): dimension of the channel.
        """

        self.register_buffer(
            "memory",
            torch.rand(
                bank_size+1,
                dim,
            )
            .to(device),
        )


    def forward(self, idx, data):

        data_averages = torch.index_select(self.memory, 0, idx.view(-1)).detach()
        new_entry = data_averages.clone()
        with torch.no_grad():

            new_entry.mul_(self.momentum)
            new_entry.add_(torch.mul(data, 1 - self.momentum))
            # what would be the reason for doing euclidean norm here?
            # TODO: SINCE january 10, we are not doing euclidean norm here
            # norm = new_entry.pow(2).sum(1, keepdim=True).pow(0.5)
            # updated = new_entry.div(norm)
            self.memory.index_copy_(0, idx, new_entry)

        return data_averages

class CLIPSim(nn.Module):
    def __init__(self, eegModel,
                 speechModel, eegMemoryBank,
                 temperature=1., latent_dim=16, window_length=192,
                 lambda_clip=1, lambda_average=1):
        super(CLIPSim, self).__init__()
        self.eegModel = eegModel
        self.speechModel = speechModel
        self.eegMemoryBank = eegMemoryBank
        self.latent_dim = latent_dim
        self.window_length = window_length
        self.lambda_clip = lambda_clip
        self.lambda_average = lambda_average

        #linear project latent space to the same dimension
        self.latent_projection_eeg = nn.Linear(self.eegModel.get_output_dim(input_window_size=self.window_length), self.latent_dim, bias=False)
        self.latent_projection_speech = nn.Linear(self.eegModel.get_output_dim(input_window_size=self.window_length), self.latent_dim, bias=False)

        self.temperature = nn.Parameter(torch.tensor(temperature))


    def forward(self, eeg,speech,  ids):
        # Getting Image and Text Features
        EEG_features = self.eegModel(eeg)
        speech_features = self.speechModel(speech)

        # Getting Image and Text Embeddings (with same dimension)
        EEG_features = torch.flatten(EEG_features, start_dim=1)
        speech_features = torch.flatten(speech_features, start_dim=1)

        # put throught the linear projection
        EEG_features = self.latent_projection_eeg(EEG_features)
        speech_features = self.latent_projection_speech(speech_features)

        # l2-normalize them
        EEG_features = F.normalize(EEG_features, p=2, dim=1)
        speech_features = F.normalize(speech_features, p=2, dim=1)

        # get the average of the EEG features from the memory bank
        EEG_features_average = self.eegMemoryBank(ids, EEG_features)
        # l2-normalize them
        EEG_features_average = F.normalize(EEG_features_average, p=2, dim=1)


        # Calculating the Loss
        logits = (speech_features @ EEG_features.T) * torch.exp(self.temperature)

        # targets but in torch
        targets = torch.arange(logits.shape[0], device=logits.device)

        # cross-enropy loss - contrastive objective
        speech_loss = F.cross_entropy(logits, targets)
        EEG_loss = F.cross_entropy(logits.T, targets.T)


        # average EEG features loss - regression objective
        EEG_average_loss = F.mse_loss(EEG_features_average, EEG_features)

        loss_ce = (speech_loss + EEG_loss) / 2.0

        loss_total = self.lambda_clip * loss_ce + self.lambda_average * EEG_average_loss


        return  loss_ce.mean(), EEG_average_loss.mean(), loss_total.mean()

    # PyTor

class BaseMatchMismatch(nn.Module):
    def __init__(self, eegModel,
                 speechModel,
                 latent_dim=16, window_length=192,
                 temperature=0.075
                 ):
        super(BaseMatchMismatch, self).__init__()
        self.eegModel = eegModel
        self.speechModel = speechModel
        self.latent_dim = latent_dim
        self.window_length = window_length
        self.temperature = nn.Parameter(torch.tensor(temperature))


    def forward(self, eeg,speech):
        # Getting Image and Text Features
        EEG_features = self.eegModel(eeg)
        speech_features = self.speechModel(speech)

        # flatten to one vector
        EEG_features = torch.flatten(EEG_features, start_dim=1)
        speech_features = torch.flatten(speech_features, start_dim=1)

        # l2-normalize them
        EEG_features = F.normalize(EEG_features, p=2, dim=1)
        speech_features = F.normalize(speech_features, p=2, dim=1)


        logits = (speech_features @ EEG_features.T) * torch.exp(self.temperature)
        logits_match = torch.diagonal(logits)
        # take the below diagonal

        logits_mismatch = torch.diagonal(logits, offset=1)
        final_mismatch = logits[-2,-1]
        logits_mismatch = torch.cat((logits_mismatch, final_mismatch.unsqueeze(0)))


        logits = torch.stack((logits_match, logits_mismatch))

        logits = torch.transpose(logits, 0, 1)

        # targets but in in a logits shape , eg (bs, 2)
        targets = torch.stack((torch.ones(logits.shape[0], dtype=torch.float, device=logits.device), torch.zeros(logits.shape[0], dtype=torch.float, device=logits.device)))
        targets = torch.transpose(targets, 0, 1)

        # cross-enropy loss - contrastive objective
        speech_loss = F.cross_entropy(logits, targets)

        # calculate the accuracy
        accuracy = (logits.argmax(1) == targets.argmax(1)).float().mean()

        return speech_loss.mean(), accuracy

#
class CLIPSimNoLatentProj(nn.Module):
    def __init__(self, eegModel,
                 speechModel, eegMemoryBank,
                 temperature=1., window_length=192,
                 lambda_clip=1, lambda_average=1):
        super(CLIPSimNoLatentProj, self).__init__()
        self.eegModel = eegModel
        self.speechModel = speechModel
        self.eegMemoryBank = eegMemoryBank

        self.window_length = window_length
        self.lambda_clip = lambda_clip
        self.lambda_average = lambda_average

        #linear project la
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.temperature_eeg = nn.Parameter(torch.tensor(temperature))


    def forward(self, eeg,speech,  ids):
        # Getting Image and Text Features
        EEG_features = self.eegModel(eeg)
        speech_features = self.speechModel(speech)

        # if speech shape 1 > speech shape 2, swap
        if speech_features.shape[1] > speech_features.shape[2]:
            speech_features = torch.transpose(speech_features, 1, 2)

        if EEG_features.shape[1] > EEG_features.shape[2]:
            EEG_features = torch.transpose(EEG_features, 1, 2)

        # ensure speech and EEG features have the same dimension, otherwise crop the speech features
        # if EEG_features.shape[1] != speech_features.shape[1]:
        #     speech_features = speech_features[:, :EEG_features.shape[1]]
        #     EEG_features = EEG_features[:, :speech_features.shape[1]]
        # # same for shape 2
        # if EEG_features.shape[2] != speech_features.shape[2]:
        #     speech_features = speech_features[:, :, :EEG_features.shape[2]]
        #     EEG_features = EEG_features[:, :, :speech_features.shape[2]]

        # Getting Image and Text Embeddings (with same dimension)
        EEG_features = torch.flatten(EEG_features, start_dim=1)
        speech_features = torch.flatten(speech_features, start_dim=1)

        # l2-normalize them
        EEG_features = F.normalize(EEG_features, p=2, dim=1)
        speech_features = F.normalize(speech_features, p=2, dim=1)

        # get the average of the EEG features from the memory bank
        EEG_features_average = self.eegMemoryBank(ids, EEG_features)
        # l2-normalize them
        EEG_features_average = F.normalize(EEG_features_average, p=2, dim=1)


        # Calculating the Loss
        logits = (speech_features @ EEG_features.T) * torch.exp(self.temperature)

        # targets but in torch
        targets = torch.arange(logits.shape[0], device=logits.device)

        # cross-enropy loss - contrastive objective
        speech_loss = F.cross_entropy(logits, targets)
        EEG_loss = F.cross_entropy(logits.T, targets.T)


        # average EEG features loss - regression objective - also do cosine similatiry between the average and the EEG features
        logits_EEG = (EEG_features_average @ EEG_features.T) * torch.exp(self.temperature_eeg)
        targets_eeg = torch.arange(logits_EEG.shape[0], device=logits_EEG.device)

        EEG_average_loss = F.cross_entropy(logits_EEG, targets_eeg)

        loss_ce = (speech_loss + EEG_loss) / 2.0

        loss_total = self.lambda_clip * loss_ce + self.lambda_average * EEG_average_loss


        return  loss_ce.mean(), EEG_average_loss.mean(), loss_total.mean()

    # PyTor

class CLIPNoContrastiveLearning(nn.Module):
    def __init__(self, eegModel,
                 speechModel,
                 window_length=192):
        super(CLIPNoContrastiveLearning, self).__init__()
        self.eegModel = eegModel
        self.speechModel = speechModel
        self.window_length = window_length



    def forward(self, eeg,speech,  ids):
        # Getting Image and Text Features
        EEG_features = self.eegModel(eeg)
        speech_features = self.speechModel(speech)

        # if speech shape 1 > speech shape 2, swap
        if speech_features.shape[1] > speech_features.shape[2]:
            speech_features = torch.transpose(speech_features, 1, 2)

        if EEG_features.shape[1] > EEG_features.shape[2]:
            EEG_features = torch.transpose(EEG_features, 1, 2)

        # Getting Image and Text Embeddings (with same dimension)
        EEG_features = torch.flatten(EEG_features, start_dim=1)
        speech_features = torch.flatten(speech_features, start_dim=1)

        # l2-normalize them
        EEG_features = F.normalize(EEG_features, p=2, dim=1)
        speech_features = F.normalize(speech_features, p=2, dim=1)

        # Calculating the Loss
        logits = (speech_features @ EEG_features.T)

        logits_match = torch.diagonal(logits)[:-1]
        # take the below diagonal
        logits_mismatch = torch.diagonal(logits, offset=1)

        # targets but in torch
        targets = torch.stack([torch.ones(logits.shape[0]-1, device=logits.device),
                               torch.zeros(logits.shape[0]-1, device=logits.device)])

        # cross-enropy loss - contrastive objective
        speech_loss = F.binary_cross_entropy_with_logits(torch.stack([logits_match,logits_mismatch]), targets)



        return  speech_loss.mean(), speech_loss.mean(), speech_loss.mean()

    # PyTor


class CLIPSimMultiplePositives(nn.Module):
    def __init__(self, eegModel,
                 speechModel,
                 temperature=1., window_length=192,
                 lambda_clip=1, lambda_average=1):
        super(CLIPSimMultiplePositives, self).__init__()
        self.eegModel = eegModel
        self.speechModel = speechModel


        self.window_length = window_length
        self.lambda_clip = lambda_clip
        self.lambda_average = lambda_average

        #linear project la
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.temperature_eeg = nn.Parameter(torch.tensor(temperature))


    def forward(self, eeg,speech,  ids):
        # Getting Image and Text Features
        EEG_features = self.eegModel(eeg)
        speech_features = self.speechModel(speech)

        # # if speech shape 1 > speech shape 2, swap
        # if speech_features.shape[1] > speech_features.shape[2]:
        #     speech_features = torch.transpose(speech_features, 1, 2)
        #
        # if EEG_features.shape[1] > EEG_features.shape[2]:
        #     EEG_features = torch.transpose(EEG_features, 1, 2)
        #
        # # ensure speech and EEG features have the same dimension, otherwise crop the speech features
        # if EEG_features.shape[1] != speech_features.shape[1]:
        #     speech_features = speech_features[:, :EEG_features.shape[1]]
        #     EEG_features = EEG_features[:, :speech_features.shape[1]]
        # # same for shape 2
        # if EEG_features.shape[2] != speech_features.shape[2]:
        #     speech_features = speech_features[:, :, :EEG_features.shape[2]]
        #     EEG_features = EEG_features[:, :, :speech_features.shape[2]]

        # Getting Image and Text Embeddings (with same dimension)
        EEG_features = torch.flatten(EEG_features, start_dim=1)
        speech_features = torch.flatten(speech_features, start_dim=1)

        # l2-normalize them
        EEG_features = F.normalize(EEG_features, p=2, dim=1)
        speech_features = F.normalize(speech_features, p=2, dim=1)

        # speech_features = speech_features[0:32,:]

        # Calculating the Loss
        logits = (speech_features @ EEG_features.T) * torch.exp(self.temperature)
        targets = torch.arange(logits.shape[0], device=logits.device)
        # copy targets x times
        n_repeats = int(logits.shape[1]/targets.shape[0])
        targets = torch.cat((targets,)*n_repeats)

        EEG_loss = F.cross_entropy(logits.T, targets.T)


        # group logits by 4
        logits_speech  = torch.reshape(logits, (logits.shape[0], -1,logits.shape[0]))
        # targets but in torch
        targets_speech = torch.arange(logits_speech.shape[0], device=logits_speech.device)
        # cross-enropy loss - contrastive objective
        speech_loss = multiple_postives_loss(logits_speech, targets_speech)

        # now we calculate the loss between the different EEG examples as an extra loss
        sim_loss = simloss(logits_speech, targets_speech)




        loss_ce = (speech_loss + EEG_loss) / 2.0

        loss_total = self.lambda_clip * loss_ce + self.lambda_average * sim_loss


        return  loss_ce.mean(), sim_loss.mean(), loss_total.mean()

    # PyTor


class CLIPSimMultiplePositivesAdapted(nn.Module):
    def __init__(self, eegModel,
                 speechModel,
                 temperature=1., window_length=192,
                 lambda_clip=1, lambda_average=1):
        super(CLIPSimMultiplePositivesAdapted, self).__init__()
        self.eegModel = eegModel
        self.speechModel = speechModel


        self.window_length = window_length
        self.lambda_clip = lambda_clip
        self.lambda_average = lambda_average

        #linear project la
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.temperature_eeg = nn.Parameter(torch.tensor(temperature))


    def forward(self, eeg,speech,  ids):
        # Getting Image and Text Features
        EEG_features = self.eegModel(eeg)
        speech_features = self.speechModel(speech)

        # # if speech shape 1 > speech shape 2, swap
        # if speech_features.shape[1] > speech_features.shape[2]:
        #     speech_features = torch.transpose(speech_features, 1, 2)
        #
        # if EEG_features.shape[1] > EEG_features.shape[2]:
        #     EEG_features = torch.transpose(EEG_features, 1, 2)
        #
        # # ensure speech and EEG features have the same dimension, otherwise crop the speech features
        # if EEG_features.shape[1] != speech_features.shape[1]:
        #     speech_features = speech_features[:, :EEG_features.shape[1]]
        #     EEG_features = EEG_features[:, :speech_features.shape[1]]
        # # same for shape 2
        # if EEG_features.shape[2] != speech_features.shape[2]:
        #     speech_features = speech_features[:, :, :EEG_features.shape[2]]
        #     EEG_features = EEG_features[:, :, :speech_features.shape[2]]

        # Getting Image and Text Embeddings (with same dimension)
        EEG_features = torch.flatten(EEG_features, start_dim=1)
        speech_features = torch.flatten(speech_features, start_dim=1)

        # l2-normalize them
        EEG_features = F.normalize(EEG_features, p=2, dim=1)
        speech_features = F.normalize(speech_features, p=2, dim=1)

        # speech_features = speech_features[0:32,:]

        # Calculating the Loss
        logits = (speech_features @ EEG_features.T) * torch.exp(self.temperature)
        targets = torch.arange(logits.shape[0], device=logits.device)
        # copy targets x times
        n_repeats = int(logits.shape[1]/targets.shape[0])
        targets = torch.cat((targets,)*n_repeats)

        EEG_loss = F.cross_entropy(logits.T, targets.T)


        # group logits by 4
        logits_speech  = torch.reshape(logits, (logits.shape[0], -1,logits.shape[0]))

        # add the logits together,
        logits_speech = torch.sum(logits_speech, dim=1)

        # targets but in torch
        targets_speech = torch.arange(logits_speech.shape[0], device=logits_speech.device)

        # cross-enropy loss - contrastive objective
        # we take the sum, such that all the logits have to be a good representation, and small, and not just one of them
        speech_loss = F.cross_entropy(logits_speech, targets_speech)
        # speech_loss = multiple_postives_loss(logits_speech, targets_speech)

        # now we calculate the loss between the different EEG examples as an extra loss
        # sim_loss = simloss(logits_speech, targets_speech)




        loss_ce = (speech_loss + EEG_loss) / 2.0

        loss_total = self.lambda_clip * loss_ce #+ self.lambda_average * sim_loss


        return  loss_ce.mean(), loss_ce.mean(), loss_total.mean()

    # PyTor



class CLIPKLDNoLatentProj(nn.Module):
    def __init__(self, eegModel,
                 speechModel, latent_dimension,
                 number_of_classes, latent_dimension2 = 64,
                 temperature=1., window_length=192,
                 lambda_clip=1, lambda_lower_bound=1, lambda_discriminative=1):
        super(CLIPKLDNoLatentProj, self).__init__()
        self.eegModel = eegModel
        self.speechModel = speechModel

        self.window_length = window_length
        self.lambda_clip = lambda_clip
        self.lambda_lower_bound = lambda_lower_bound
        self.lambda_discriminative = lambda_discriminative
        self.number_of_classes = number_of_classes
        self.latent_dimension2 = latent_dimension2

        #linear project la
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.temperature_eeg = nn.Parameter(torch.tensor(temperature))

        self.latent_dimension = latent_dimension
        self.mu_eeg_lookup = nn.Embedding(number_of_classes+1, self.latent_dimension2)
        # self.eeg_mu_linear = nn.Conv1d(self.latent_dimension, self.latent_dimension, 1)
        # self.eeg_logvar_linear = nn.Conv1d(self.latent_dimension, self.latent_dimension, 1)

        # we will apply this on the already flattened data
        self.eeg_mu_linear = nn.Linear(self.latent_dimension, self.latent_dimension2)
        self.eeg_logvar_linear = nn.Linear(self.latent_dimension, self.latent_dimension2)


    def encode(self, eeg, speech, ids):
        # Getting Image and Text Features
        EEG_features = self.eegModel(eeg)
        speech_features = self.speechModel(speech)

        # Getting Image and Text Embeddings (with same dimension)
        EEG_features = torch.flatten(EEG_features, start_dim=1)
        speech_features = torch.flatten(speech_features, start_dim=1)


        # Here we lookup the mu2 and logvar2
        mu2_eeg = self.mu_eeg_lookup(ids)
        z2_mu = self.eeg_mu_linear(EEG_features)
        z2_logvar = self.eeg_logvar_linear(EEG_features)

        # reparameterize to make sure we can backpropagate through the latent space
        z2_sample = self.reparameterize(z2_mu, z2_logvar)

        return mu2_eeg, z2_mu, z2_logvar, z2_sample, speech_features, EEG_features

    def forward(self, eeg,speech,  ids):

        # none of these are normalized yet
        mu2_eeg, z2_mu, z2_logvar, z2_sample, speech_features, EEG_features = self.encode(eeg, speech, ids)

        # Calculating the Loss

        # priors
        prior_z2 = [mu2_eeg, torch.FloatTensor([np.log(0.5 ** 2)]).to(z2_mu.device)]
        prior_mu2 = [torch.FloatTensor([0]).to(z2_mu.device), torch.FloatTensor([np.log(1 ** 2)]).to(z2_mu.device)]

        # variational lower bound
        log_pmu2 = torch.mean(log_gauss(mu2_eeg, prior_mu2[0], prior_mu2[1]), dim=1)
        kld_z2 = torch.mean(kld(z2_mu, z2_logvar, prior_z2[0], prior_z2[1]), dim=1)

        lower_bound = torch.mean(-log_pmu2 + kld_z2, dim=0)

        # discriminative loss - see if necesary
        # logits = z2_sample.unsqueeze(1) - mu2_eeg.unsqueeze(0)
        # logits = -1 * torch.pow(logits, 2) / (2 * torch.exp(prior_z2[1]))
        # logits = torch.mean(logits, dim=-1)
        #
        # log_qy = F.cross_entropy(logits, ids)



        # CLIP Loss - contrastive objective

        # l2-normalize them
        EEG_features = F.normalize(EEG_features, p=2, dim=1)
        speech_features = F.normalize(speech_features, p=2, dim=1)

        logits = (speech_features @ EEG_features.T) * torch.exp(self.temperature)

        # targets but in torch
        targets = torch.arange(logits.shape[0], device=logits.device)

        # cross-enropy loss - contrastive objective
        speech_loss = F.cross_entropy(logits, targets)
        EEG_loss = F.cross_entropy(logits.T, targets.T)
        loss_ce = (speech_loss + EEG_loss) / 2.0

        loss_total = self.lambda_clip * loss_ce + self.lambda_lower_bound * lower_bound #+ self.lambda_discriminative * log_qy


        return loss_total.mean(),   loss_ce.mean(), log_pmu2.mean(), kld_z2.mean()#, log_qy.mean()


    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu


class ProjectionHead(nn.Module):
    def __init__(
            self,
            embedding_dim,
            projection_dim=512,
            dropout=0.2
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.relu = nn.LeakyReLU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.relu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class ProjectionHeadLinear(nn.Module):
    def __init__(
            self,
            embedding_dim,
            projection_dim=512,

    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim*2)
        self.relu = nn.LeakyReLU()
        self.last_linear = nn.Linear(projection_dim*2, projection_dim)


    def forward(self, x):
        projected = self.projection(x)
        x = self.relu(projected)
        x = self.last_linear(x)
        return x

class CLIPKLDWithLatentProj(nn.Module):
    def __init__(self, eegModel,
                 speechModel, latent_dimension,
                 number_of_classes,
                 temperature=1., window_length=192,
                 lambda_clip=1, lambda_lower_bound=1, lambda_discriminative=1,
                 projection_head='linear'):
        super(CLIPKLDWithLatentProj, self).__init__()
        self.eegModel = eegModel
        self.speechModel = speechModel

        self.window_length = window_length
        self.lambda_clip = lambda_clip
        self.lambda_lower_bound = lambda_lower_bound
        self.lambda_discriminative = lambda_discriminative
        self.number_of_classes = number_of_classes

        #linear project la
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.temperature_eeg = nn.Parameter(torch.tensor(temperature))

        self.latent_dimension = latent_dimension
        self.mu_eeg_lookup = nn.Embedding(number_of_classes+1, self.latent_dimension)
        # self.eeg_mu_linear = nn.Conv1d(self.latent_dimension, self.latent_dimension, 1)
        # self.eeg_logvar_linear = nn.Conv1d(self.latent_dimension, self.latent_dimension, 1)

        if projection_head == 'non-linear':
            self.eeg_mu_linear = ProjectionHead(self.eegModel.get_output_dim(window_length), self.latent_dimension)
            self.eeg_logvar_linear = ProjectionHead(self.eegModel.get_output_dim(window_length), self.latent_dimension)
            self.speech_latent_projection = ProjectionHead(self.speechModel.get_output_dim(window_length), self.latent_dimension)
        else:
            # we will apply this on the already flattened data
            self.eeg_mu_linear = ProjectionHeadLinear(self.eegModel.get_output_dim(window_length), self.latent_dimension)
            self.eeg_logvar_linear = ProjectionHeadLinear(self.eegModel.get_output_dim(window_length), self.latent_dimension)
            self.speech_latent_projection = ProjectionHeadLinear(self.speechModel.get_output_dim(window_length), self.latent_dimension)



    def encode(self, eeg, speech, ids):
        # Getting Image and Text Features
        EEG_features = self.eegModel(eeg)
        speech_features = self.speechModel(speech)

        # # if speech shape 1 > speech shape 2, swap
        # if speech_features.shape[1] > speech_features.shape[2]:
        #     speech_features = torch.transpose(speech_features, 1, 2)
        #
        # # ensure speech and EEG features have the same dimension, otherwise crop the speech features
        # if EEG_features.shape[1] != speech_features.shape[1]:
        #     speech_features = speech_features[:, :EEG_features.shape[1]]
        #     EEG_features = EEG_features[:, :speech_features.shape[1]]
        # # same for shape 2
        # if EEG_features.shape[2] != speech_features.shape[2]:
        #     speech_features = speech_features[:, :, :EEG_features.shape[2]]
        #     EEG_features = EEG_features[:, :, :speech_features.shape[2]]

        # Getting Image and Text Embeddings (with same dimension)
        EEG_features = torch.flatten(EEG_features, start_dim=1)
        speech_features = torch.flatten(speech_features, start_dim=1)

        # getting the latent projections
        EEG_logvar = self.eeg_logvar_linear(EEG_features)  # logvar of the latent space
        EEG_features = self.eeg_mu_linear(EEG_features) # mean of the latent space
        speech_features = self.speech_latent_projection(speech_features)

        # l2-normalize them
        EEG_features_norm = F.normalize(EEG_features, p=2, dim=1)
        speech_features_norm = F.normalize(speech_features, p=2, dim=1)

        # Here we lookup the mu2 and logvar2
        mu2_eeg = self.mu_eeg_lookup(ids)

        # reparameterize to make sure we can backpropagate through the latent space
        z2_sample = self.reparameterize(EEG_features, EEG_logvar)

        return mu2_eeg, EEG_features, EEG_logvar, z2_sample, speech_features_norm, EEG_features_norm

    def forward(self, eeg,speech,  ids):

        mu2_eeg, z2_mu, z2_logvar, z2_sample, speech_features, EEG_features = self.encode(eeg, speech, ids)

        # Calculating the Loss

        # priors
        prior_z2 = [mu2_eeg, torch.FloatTensor([np.log(0.5 ** 2)]).to(z2_mu.device)]
        prior_mu2 = [torch.FloatTensor([0]).to(z2_mu.device), torch.FloatTensor([np.log(1.0 ** 2)]).to(z2_mu.device)]

        # variational lower bound
        log_pmu2 = torch.mean(log_gauss(mu2_eeg, prior_mu2[0], prior_mu2[1]), dim=1)
        kld_z2 = torch.mean(kld(z2_mu, z2_logvar, prior_z2[0], prior_z2[1]), dim=1)

        lower_bound = torch.mean(-log_pmu2 + kld_z2, dim=0)

        # discriminative loss - see if necesary
        # logits = z2_sample.unsqueeze(1) - mu2_eeg.unsqueeze(0)
        # logits = -1 * torch.pow(logits, 2) / (2 * torch.exp(prior_z2[1]))
        # logits = torch.mean(logits, dim=-1)
        #
        # log_qy = F.cross_entropy(logits, ids)



        # CLIP Loss - contrastive objective
        logits = (speech_features @ EEG_features.T) * torch.exp(self.temperature)

        # targets but in torch
        targets = torch.arange(logits.shape[0], device=logits.device)

        # cross-enropy loss - contrastive objective
        speech_loss = F.cross_entropy(logits, targets)
        EEG_loss = F.cross_entropy(logits.T, targets.T)
        loss_ce = (speech_loss + EEG_loss) / 2.0

        loss_total = self.lambda_clip * loss_ce + self.lambda_lower_bound * lower_bound #+ self.lambda_discriminative * log_qy


        return loss_total.mean(),   loss_ce.mean(), log_pmu2.mean(), kld_z2.mean()#, log_qy.mean()


    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu


#
# def cross_entropy(preds, targets, reduction='none'):
#     log_softmax = nn.LogSoftmax(dim=-1)
#     loss = (-targets * log_softmax(preds)).sum(1)
#     if reduction == "none":
#         return loss
#     elif reduction == "mean":
#         return loss.mean()
#

def simloss(x, target):
    logits_sum = x.sum(-2)
    return F.nll_loss(logits_sum, target)


def simloss(x, target):
    logits_sum = x.sum(-2)
    return F.nll_loss(logits_sum, target)


def log_softmax_mp(x):
    denominator = x.exp().sum(-2).sum(-1).log().unsqueeze(-1)
    nominator = x.exp().sum(-2).log()
    return nominator - denominator


def multiple_postives_loss(preds, targets, reduction='mean'):
    pred = log_softmax_mp(preds)
    loss = F.nll_loss(pred, targets, reduction=reduction)

    return loss


def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

def log_gauss(x, mu, logvar):
    log_2pi = torch.FloatTensor([np.log(2 * np.pi)]).to(x.device)
    return -0.5 * (log_2pi + logvar + torch.pow(x - mu, 2) / torch.exp(logvar))

def kld(p_mu, p_logvar, q_mu, q_logvar):
    return -0.5 * (1 + p_logvar - q_logvar - (torch.pow(p_mu - q_mu, 2) + torch.exp(p_logvar)) / torch.exp(q_logvar))

class FCNN(nn.Module):
    def __init__(self, num_hidden=1, dropout_rate=0.3, input_length=50, num_input_channels=63):
        super().__init__()

        self.input_length = input_length
        self.num_input_channels = num_input_channels

        self.num_hidden = num_hidden
        units = np.round(np.linspace(1, self.input_length * self.num_input_channels, self.num_hidden + 2)[::-1]).astype(
            int)
        self.fully_connected = torch.nn.ModuleList(
            [torch.nn.Linear(units[i], units[i + 1]) for i in range(len(units) - 1)])
        self.activations = torch.nn.ModuleList([torch.nn.Tanh() for i in range(len(units) - 2)])
        self.dropouts = torch.nn.ModuleList([torch.nn.Dropout(p=dropout_rate) for i in range(len(units) - 2)])

    def __str__(self):
        return 'fcnn'

    def forward(self, x):
        x = x.flatten(start_dim=1)
        for i, layer in enumerate(self.fully_connected[:-1]):
            x = layer(x)
            x = self.activations[i](x)
            x = self.dropouts[i](x)

        x = self.fully_connected[-1](x)
        return x.flatten()


class CNN(nn.Module):

    def __init__(self,
                 F1=16,
                 D=16,
                 F2=16,
                 dropout_rate=0.25,
                 input_length=50,
                 num_input_channels=63):
        super().__init__()

        self.input_length = input_length
        self.num_input_channels = num_input_channels

        self.F1 = F1
        self.F2 = F2
        self.D = D

        self.conv1_kernel = 3
        self.conv3_kernel = 3
        self.temporalPool1 = 2
        self.temporalPool2 = 5

        # input shape is [1, C, T]

        self.conv1 = torch.nn.Conv2d(1, self.F1, (1, self.conv1_kernel), padding='same')
        self.conv2 = torch.nn.Conv2d(self.F1, self.F1 * self.D, (self.num_input_channels, 1), padding='valid',
                                     groups=self.F1)
        self.conv3 = torch.nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, self.conv1_kernel), padding='same',
                                     groups=self.F1 * self.D)
        self.conv4 = torch.nn.Conv2d(self.F1 * self.D, self.F2, (1, 1))

        self.pool1 = torch.nn.AvgPool2d((1, self.temporalPool1))
        self.pool2 = torch.nn.AvgPool2d((1, self.temporalPool2))

        self.linear = torch.nn.Linear(self.F2 * self.input_length // (self.temporalPool1 * self.temporalPool2), 1)

        self.bnorm1 = torch.nn.BatchNorm2d(self.F1)
        self.bnorm2 = torch.nn.BatchNorm2d(self.F1 * self.D)
        self.bnorm3 = torch.nn.BatchNorm2d(F2)

        self.dropout1 = torch.nn.Dropout2d(dropout_rate)
        self.dropout2 = torch.nn.Dropout2d(dropout_rate)

        self.activation1 = torch.nn.ELU()
        self.activation2 = torch.nn.ELU()
        self.activation3 = torch.nn.ELU()

    def __str__(self):
        return 'cnn'

    def forward(self, x):
        # x shape = [batch, C, T]
        x = x.unsqueeze(1)
        out = self.conv1(x)
        out = self.bnorm1(out)

        out = self.conv2(out)
        out = self.bnorm2(out)
        out = self.activation1(out)
        out = self.pool1(out)
        out = self.dropout1(out)

        # shape is now [batch, DxF1, 1, T//TPool1]
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.bnorm3(out)
        out = self.activation2(out)
        out = self.pool2(out)
        out = self.dropout2(out)

        out = torch.flatten(out, start_dim=1)  # shape is now [batch, F2*T//(TPool1*TPool2)]
        out = self.linear(out)
        return out.flatten()