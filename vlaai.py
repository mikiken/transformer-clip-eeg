import torch
import torch.nn as nn
import torch.nn.functional as F

class Extractor(nn.Module):
    def __init__(
        self,
        filters=(256, 256, 256, 128, 128),
        kernels=(64,) * 5,
        dilation_rate=1,
        input_channels=64,
        time_dimension=64*5,
        normalization_fn='layer_norm',
        activation_fn='leaky_relu',
    ):
        super(Extractor, self).__init__()

        self.eeg = nn.Conv1d(input_channels, input_channels, kernel_size=1)  # Identity mapping for eeg

        if len(filters) != len(kernels):
            raise ValueError("'filters' and 'kernels' must have the same length")


        layers = []
        for filter_, kernel in zip(filters, kernels):

            dilation = dilation_rate

            layers.append(nn.Conv1d(in_channels= input_channels,out_channels= filter_, kernel_size=kernel, padding='same', dilation=dilation))
            if normalization_fn == 'layer_norm':
                layers.append(nn.LayerNorm([filter_, time_dimension]))


            if activation_fn == 'leaky_relu':
                layers.append(nn.LeakyReLU())

            # layers.append(nn.ConstantPad1d( padding=padding, value=0 ))

            input_channels = filter_

        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.eeg(x)
        x = self.conv_layers(x)
        return x

class OutputContext(nn.Module):
    def __init__(
        self,
        filter_=64,
        kernel=64,
        input_channels=64,
        time_dimension=64 * 5,
        normalization_fn='layer_norm',
        activation_fn='leaky_relu',
    ):
        super(OutputContext, self).__init__()

        self.conv1d = nn.Conv1d(input_channels, filter_, kernel_size=kernel, padding='same')
        if normalization_fn == 'layer_norm':
            self.normalization_fn = nn.LayerNorm([filter_, time_dimension])
        if activation_fn == 'leaky_relu':
            self.activation_fn = nn.LeakyReLU()


    def forward(self, x):
        # x = F.pad(x, (self.conv1d.padding[0], 0))
        x = self.conv1d(x)
        x = self.normalization_fn(x)
        x = self.activation_fn(x)
        return x

class VLAAI(nn.Module):
    def __init__(
        self,
        nb_blocks=4,
        extractor_model=None,
        output_context_model=None,
        use_skip=True,
        input_channels=64,
        output_dim=64,
    ):
        super(VLAAI, self).__init__()

        if extractor_model is None:
            extractor_model = Extractor()
        if output_context_model is None:
            output_context_model = OutputContext()

        linear_recombination = nn.Conv1d(in_channels=128, out_channels=input_channels, kernel_size=1,
                                         padding='same')  # recombination of 128 features to 64

        self.eeg = nn.Conv1d(input_channels, input_channels, kernel_size=1)  # Identity mapping for eeg

        if use_skip:
            self.use_skip = True # = nn.Parameter(torch.zeros(1, input_channels,1), requires_grad=False)
        else:
            self.use_skip = False

        self.sequentialConvStack= nn.Sequential(extractor_model, linear_recombination, output_context_model)
        self.output_dim = output_dim
        self.nb_blocks = nb_blocks
        self.final_linear = nn.Conv1d(input_channels, output_dim, kernel_size=1, padding='same')

    def get_output_dim(self, input_window_size):
        return input_window_size * self.output_dim

    def forward(self, x):

        # change dimensions of x
        x = x.transpose(1, 2)

        if self.use_skip:
            eeg = x
        else:
            # get shape of x
            eeg = nn.Parameter(torch.zeros(1, x.shape[1],1), requires_grad=False)

        eeg.to(x.device)


        x = self.eeg(x)
        for idx in range(self.nb_blocks):

            if idx == 0 or idx == self.nb_blocks - 1:

                x = self.sequentialConvStack(x)
            else:
                x = self.sequentialConvStack(x+eeg )

        x = self.final_linear(x)

        return x