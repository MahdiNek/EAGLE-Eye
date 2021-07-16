
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import torch
import torch.nn
import torchvision

from torch.nn import Module, Conv3d, BatchNorm3d, MaxPool3d, ReLU
from torch.nn import functional as F
from scipy import signal

from AQA_head.nets.utils import ChannelShuffleLayer, Channelwise_1D, TemporalConv1DLayer, MaxPoolChannel, AvgPoolChannel



class AQA(Module):
    """
    AQA implementation in pytorch
    """

    def __init__(self, input_shape_JCA, input_shape_ADA, n_layers_JCA=4,
                 n_layers_ADA=3,n_branches_JCA=3, n_branches_ADA=3,
                 expansion_factor_JCA = 1.25, expansion_factor_ADA = 1.25, final_expansion_JCA_for_balance = 10,
                 figure_skating_flag = 0, Ablated_flag = 'N',
                 Spatial_Attention_Method = 'N', Temporal_Attention_Method = 'N',
                 Coeff_Spatial_Attention = 1, Coeff_Temporal_Attention=1):

        super(AQA, self).__init__()
        is_dilated = False

        self.expansion_factor_JCA = expansion_factor_JCA
        self.expansion_factor_ADA = expansion_factor_ADA
        self.n_layers_JCA = n_layers_JCA
        self.n_layers_ADA = n_layers_ADA
        self.is_dilated = is_dilated
        self.n_branches_JCA = n_branches_JCA
        self.n_branches_ADA = n_branches_ADA
        self.final_expansion_JCA_for_balance = final_expansion_JCA_for_balance
        self.figure_skating_flag = figure_skating_flag
        self.n_channels_out = None
        self.Ablated_flag = Ablated_flag
        self.Spatial_Attention_Method = Spatial_Attention_Method
        self.Temporal_Attention_Method = Temporal_Attention_Method
        self.Coeff_Spatial_Attention = Coeff_Spatial_Attention
        self.Coeff_Temporal_Attention = Coeff_Temporal_Attention


        input_shape_JCA = list(input_shape_JCA)
        input_shape_ADA = list(input_shape_ADA)
        n_channels_in_ADA = input_shape_ADA[1]
        n_channels_in_JCA = input_shape_JCA[1]

        n_channels_out_JCA = self.__define_JCA_layers_overall(input_shape_JCA, n_layers_JCA, n_branches_JCA,
                                                      expansion_factor_JCA, is_dilated)
        n_channels_out_ADA = self.__define_ADA_layers_overall(input_shape_ADA, n_layers_ADA, n_branches_ADA,
                                                      expansion_factor_ADA, is_dilated)

        self.n_channels_out_JCA = n_channels_out_JCA
        self.n_channels_out_ADA = n_channels_out_ADA

        if self.Ablated_flag == 'ADA':
            self.n_channels_out = self.n_channels_out_ADA
        elif self.Ablated_flag == 'JCA':
            self.n_channels_out = self.n_channels_out_JCA * final_expansion_JCA_for_balance
        elif self.Ablated_flag == 'Appearance':
            self.n_channels_out = n_channels_in_ADA
        elif self.Ablated_flag == 'Pose':
            self.n_channels_out = n_channels_in_JCA
        else:
            self.n_channels_out = (self.n_channels_out_ADA + self.n_channels_out_JCA * final_expansion_JCA_for_balance)

    def forward(self, input):

        input_ADA = input[0]
        input_JCA = input[1]
        input_ADA = self.__attention_module(input_ADA)
        input_JCA = self.__attention_module(input_JCA)

        expansion_factor_JCA = self.expansion_factor_JCA
        expansion_factor_ADA = self.expansion_factor_ADA
        n_layers_JCA = self.n_layers_JCA
        n_layers_ADA = self.n_layers_ADA
        n_branches_JCA = self.n_branches_JCA
        n_branches_ADA = self.n_branches_ADA

        out = []

        if self.Ablated_flag == 'ADA':
            output_ADA = self.__call_ADA_layers(input_ADA, n_layers_ADA, n_branches_ADA, expansion_factor_ADA)
            out.append(output_ADA)
            output = output_ADA
        elif self.Ablated_flag == 'JCA':
            output_JCA = self.__call_JCA_layers(input_JCA, n_layers_JCA, n_branches_JCA, expansion_factor_JCA)
            out.append(output_JCA)
            output = output_JCA
        elif self.Ablated_flag == 'Appearance':
            output = input[0]
        elif self.Ablated_flag == 'Pose':
            output = input[1]
        else:
            output_ADA = self.__call_ADA_layers(input_ADA, n_layers_ADA, n_branches_ADA, expansion_factor_ADA)
            out.append(output_ADA)
            output_JCA = self.__call_JCA_layers(input_JCA, n_layers_JCA, n_branches_JCA, expansion_factor_JCA)
            out.append(output_JCA)
            output = torch.cat(out, dim=1)

        return output

    def __define_JCA_layers_overall(self, input_shape_JCA, n_layers_JCA, n_branches_JCA, expansion_factor_JCA, is_dilated):
        """
        Define Overall JCA layers (Not the details)
        """

        # how many layers of timeception
        for i in range(n_layers_JCA):

            n_channels_in_JCA = input_shape_JCA[1]
            layer_num = i + 1

            # get details about grouping
            getchannels = self.__get_n_channels_for_JCA(expansion_factor_JCA, n_branches_JCA, n_channels_in_JCA)
            n_channels_per_branch_in_JCA, n_channels_base_out_JCA, n_channels_out_JCA = getchannels


            # temporal conv per group
            self.__define_JCA(input_shape_JCA, n_branches_JCA, is_dilated, layer_num)

            # activation
            layer_name = 'relu_JCA%d' % (layer_num)
            layer = ReLU()
            layer._name = layer_name
            setattr(self, layer_name, layer)

            # SpatialConvForHeatmaps
            layer_name = 'conv_spatial_JCA%d' % (layer_num)
            layer = Conv3d(n_channels_out_JCA , n_channels_out_JCA , kernel_size=(1, 3, 3),padding=(0,layer_num!=n_layers_JCA,layer_num!=n_layers_JCA))
            layer._name = layer_name
            setattr(self, layer_name, layer)

            layer_name = 'maxpool_spatial_JCA%d' % (layer_num)
            layer = MaxPool3d(kernel_size=(1, (self.figure_skating_flag != 1) + 1, (self.figure_skating_flag != 1) + 1))
            layer._name = layer_name
            setattr(self, layer_name, layer)

            n_channels_in_JCA = n_channels_out_JCA
            input_shape_JCA[1] = n_channels_in_JCA
            input_shape_JCA[2] = int(input_shape_JCA[2]/float(2))

        layer_name = 'temp&spatialMaxPool_JCA_last_adapt'
        layer = MaxPool3d(kernel_size=(8, (self.n_layers_JCA != 3 or self.figure_skating_flag==1) + 1, (self.n_layers_JCA != 3 or self.figure_skating_flag==1) + 1))
        layer._name = layer_name
        setattr(self, layer_name, layer)

        layer_name = 'channel_expansion_JCA_last_adapt'
        layer = Conv3d(input_shape_JCA[1], input_shape_JCA[1] * self.final_expansion_JCA_for_balance , kernel_size=(1, 1, 1))
        layer._name = layer_name
        setattr(self, layer_name, layer)

        return n_channels_in_JCA

    def __define_JCA(self, input_shape_JCA,n_branches_JCA, is_dilated, layer_num):
        """
        Define detailed covolutions inside the each JCA block
        """
        getchannels_JCA = self.__get_n_channels_for_JCA(self.expansion_factor_JCA, n_branches_JCA, input_shape_JCA[1])
        n_channels_per_branch_in_JCA, n_channels_base_branch_out_JCA, n_channels_out_JCA = getchannels_JCA

        #assert n_channels_in_JCA % n_channels_per_branch_in_JCA == 0

        # type of multi-scale kernels to use: either multi_kernel_sizes or multi_dilation_rates
        if is_dilated:
            kernel_sizes_temporal = (3, 3, 3)
            dilation_rates_temporal = (1, 2, 3)
        else:
            kernel_sizes_temporal = (3, 5, 7)
            dilation_rates_temporal = (1, 1, 1)

        kernel_size_channel = 3
        dilation_rate_channel = 1

        for branch_num in range (n_branches_JCA):

            base_AvgPool_size = int(n_channels_per_branch_in_JCA/float(n_channels_base_branch_out_JCA))
            n_channels_current_branch_out_JCA = int(n_channels_per_branch_in_JCA/(base_AvgPool_size + branch_num))

            ## Temporal
            layer_name = 'temporal_b%d_JCA%d' % (branch_num, layer_num)
            layer = TemporalConv1DLayer(input_shape_JCA, kernel_sizes_temporal[branch_num],
                                   dilation_rates_temporal[branch_num], layer_name)
            setattr(self, layer_name, layer)

            ## Temporal MaxPool
            layer_name = 'tmpmaxpool_b%d_JCA%d' % (branch_num, layer_num)
            layer = MaxPool3d(kernel_size=(2,1,1))
            layer._name = layer_name
            setattr(self, layer_name, layer)

            ## Channelwise
            layer_name = 'convch_b%d_JCA%d' % (branch_num, layer_num)
            layer = Channelwise_1D(input_shape_JCA, kernel_size_channel,
                                   dilation_rate_channel, layer_name)
            setattr(self, layer_name, layer)


            ## Channel-wise AvgPool
            # ChannelPoolsize = int(n_channels_per_branch_in_JCA/n_channels_per_branch_out_JCA)

            layer_name = 'chavgpool_b%d_JCA%d' % (branch_num, layer_num)
            layer = AvgPoolChannel(base_AvgPool_size + branch_num, layer_name)
            layer._name = layer_name
            setattr(self, layer_name, layer)

            ## BatchNorm
            layer_name = 'bn1_b%d_JCA%d' % (branch_num, layer_num)
            layer = BatchNorm3d(n_channels_current_branch_out_JCA)
            layer._name = layer_name
            setattr(self, layer_name, layer)


            ## BatchNorm
            # layer_name = 'bn2_b%d_JCA%d' % (branch_num, layer_num)
            # layer = BatchNorm3d(n_channels_current_branch_out_JCA)
            # layer._name = layer_name
            # setattr(self, layer_name, layer)


    def __define_ADA_layers_overall(self, input_shape_ADA, n_layers_ADA, n_branches_ADA, expansion_factor_ADA, is_dilated):
        """
        Define layers inside the timeception layers.
        """

        # how many layers of timeception
        for i in range(n_layers_ADA):

            n_channels_in_ADA = input_shape_ADA[1]

            layer_num = i + 1

            # get details about grouping
            getchannels_ADA = self.__get_n_channels_for_ADA(expansion_factor_ADA, n_branches_ADA, n_channels_in_ADA=
                                                                            n_channels_in_ADA)
            n_channels_out_sep_per_branch_in_out, n_channels_out_ADA = getchannels_ADA

            # channel reduction
            layer_name = 'chreduce_ADA%d' % (layer_num)
            layer = Conv3d(n_channels_in_ADA , n_channels_out_sep_per_branch_in_out , kernel_size=(1, 1, 1))
            layer._name = layer_name
            setattr(self, layer_name, layer)

            # temporal conv per group
            self.__define_ADA(input_shape_ADA, n_branches_ADA, is_dilated, layer_num)

            # activation
            layer_name = 'relu_ADA%d' % (layer_num)
            layer = ReLU()
            layer._name = layer_name
            setattr(self, layer_name, layer)

            # SpatialConv
            if self.Spatial_Attention_Method != 'N':

                layer_name = 'conv_spatial_ADA%d' % (layer_num)
                layer = Conv3d(n_channels_out_ADA , n_channels_out_ADA , kernel_size=(1, 3, 3), padding= (0,1,1))
                layer._name = layer_name
                setattr(self, layer_name, layer)


            n_channels_in_ADA = n_channels_out_ADA
            input_shape_ADA[1] = n_channels_in_ADA


        return n_channels_in_ADA


    def __define_ADA(self, input_shape_ADA, n_branches_ADA, is_dilated, layer_num):
        """
        Define layers inside grouped convolutional block.
        """

        getchannels_ADA = self.__get_n_channels_for_ADA(self.expansion_factor_ADA, n_branches_ADA, input_shape_ADA[1])
        n_channels_sep_per_branch_in_out_ADA, n_channels_out_ADA = getchannels_ADA

        #assert n_channels_in_ADA % n_channels_per_branch_in_ADA == 0

        # type of multi-scale kernels to use: either multi_kernel_sizes or multi_dilation_rates
        if is_dilated:
            kernel_sizes_temporal = (3, 3, 3)
            dilation_rates_temporal = (1, 2, 3)
        else:
            kernel_sizes_temporal = (3, 5, 7)
            dilation_rates_temporal = (1, 1, 1)


        for branch_num in range (n_branches_ADA):

            ## Temporal
            temp_inp_branch_ADA = input_shape_ADA
            temp_inp_branch_ADA[1] = n_channels_sep_per_branch_in_out_ADA
            layer_name = 'temporal_b%d_ADA%d' % (branch_num, layer_num)
            layer = TemporalConv1DLayer(temp_inp_branch_ADA, kernel_sizes_temporal[branch_num],
                                   dilation_rates_temporal[branch_num], layer_name)

            setattr(self, layer_name, layer)

            ## BatchNorm
            layer_name = 'bn1_b%d_ADA%d' % (branch_num, layer_num)
            layer = BatchNorm3d(n_channels_sep_per_branch_in_out_ADA)
            layer._name = layer_name
            setattr(self, layer_name, layer)

            ## Temporal MaxPool
            layer_name = 'tmpmaxpool_b%d_ADA%d' % (branch_num, layer_num)
            layer = MaxPool3d(kernel_size=(2,1,1))
            layer._name = layer_name
            setattr(self, layer_name, layer)

            ## BatchNorm
            layer_name = 'bn2_b%d_ADA%d' % (branch_num, layer_num)
            layer = BatchNorm3d(n_channels_sep_per_branch_in_out_ADA)
            layer._name = layer_name
            setattr(self, layer_name, layer)

    def __call_JCA_layers(self, tensor, n_layers_JCA, n_branches_JCA, expansion_factor_JCA):


        input_shape = tensor.size()
        n_channels_in_JCA = input_shape[1]

        # how many layers of timeception

        for i in range(n_layers_JCA):
            layer_num = i + 1

            # get details about grouping
            getchannels_JCA = self.__get_n_channels_for_JCA(expansion_factor_JCA, n_branches_JCA, n_channels_in_JCA)
            n_channels_per_branch_in, n_channels_base_branch_out, n_channels_out = getchannels_JCA

            # temporal conv per group
            tensor = self.__call_JCA(tensor, layer_num, n_branches_JCA)

            n_channels_in_JCA = n_channels_out

        tensor = getattr(self, 'temp&spatialMaxPool_JCA_last_adapt')(tensor)
        tensor = getattr(self, 'channel_expansion_JCA_last_adapt')(tensor)


        return tensor


    def __call_JCA(self, tensor, layer_num, n_branches):

        t = []

        for branches in range (n_branches):

            t_1 = getattr(self, 'temporal_b%d_JCA%d' % (branches, layer_num))(tensor)
            t_2 = getattr(self, 'convch_b%d_JCA%d' % (branches, layer_num))(t_1)
            t_3 = getattr(self, 'chavgpool_b%d_JCA%d' % (branches, layer_num))(t_2)
            t_4 = getattr(self, 'tmpmaxpool_b%d_JCA%d' % (branches, layer_num))(t_3)
            t_5 = getattr(self, 'bn1_b%d_JCA%d' % (branches, layer_num))(t_4)
            t.append(t_5)

        tensor = torch.cat(t, dim=1)
        tensor = getattr(self, 'relu_JCA%d' % (layer_num))(tensor)

        tensor = getattr(self, 'conv_spatial_JCA%d' % (layer_num))(tensor)
        tensor = getattr(self, 'maxpool_spatial_JCA%d' % (layer_num))(tensor)



        # concatenate channels of branches

        return tensor

    def __call_ADA_layers(self, tensor, n_layers_ADA, n_branches_ADA, expansion_factor_ADA):


        input_shape = tensor.size()
        n_channels_in_ADA = input_shape[1]

        # how many layers of timeception

        for i in range(n_layers_ADA):
            layer_num = i + 1

            # get details about grouping
            getchannels_ADA = self.__get_n_channels_for_ADA(expansion_factor_ADA, n_branches_ADA, n_channels_in_ADA )
            n_channels_sep_per_branch_in_out, n_channels_out = getchannels_ADA

            # temporal conv per group
            tensor = self.__call_ADA(tensor, layer_num, n_branches_ADA)

            n_channels_in_ADA = n_channels_out


        return tensor


    def __call_ADA(self, tensor, layer_num, n_branches):

        t_1 = getattr(self, 'chreduce_ADA%d' % (layer_num))(tensor)
        t = []

        for branches in range (n_branches):

            t_2 = getattr(self, 'temporal_b%d_ADA%d' % (branches, layer_num))(t_1)
            # t_3 = getattr(self, 'bn1_b%d_ADA%d' % (branches, layer_num))(t_2)
            t_3 = getattr(self, 'tmpmaxpool_b%d_ADA%d' % (branches, layer_num))(t_2)
            t_5 = getattr(self, 'bn2_b%d_ADA%d' % (branches, layer_num))(t_3)
            t.append(t_5)

        t = torch.cat(t, dim=1)
        tensor = getattr(self, 'relu_ADA%d' % (layer_num))(t)

        if self.Spatial_Attention_Method != 'N':
            tensor = getattr(self, 'conv_spatial_ADA%d' % (layer_num))(tensor)

        # concatenate channels of branches

        return tensor

    def __get_n_channels_for_JCA(self, expansion_factor_JCA, n_branches_JCA, n_channels_in_JCA):

        if n_branches_JCA == 3:
            if expansion_factor_JCA == 13/12:
                channelAvgPool_size = 2
            elif expansion_factor_JCA == 47/60:
                channelAvgPool_size = 3
            else:
                raise ValueError('Current setting of expansion factor is not practical. Please try again')
        elif n_branches_JCA == 2:
            if expansion_factor_JCA == 5 / 6:
                channelAvgPool_size = 2
            elif expansion_factor_JCA == 7/12:
                channelAvgPool_size = 3
            else:
                raise ValueError('Current setting of expansion factor is not practical. Please try again')
        else:
            raise ValueError('Deploying more than 3 branches or a single branch is not valid. Please try again')

        n_channels_per_branch_in = int(n_channels_in_JCA)
        #  n_channels_base_branch_out = int(n_channels_in_JCA * expansion_factor_JCA / float(n_branches_JCA))
        n_channels_base_branch_out = int (n_channels_in_JCA/float(channelAvgPool_size))

        n_channels_out = 0
        for branch in range(n_branches_JCA):
            n_channels_out = n_channels_out + int(n_channels_in_JCA/float(channelAvgPool_size + branch))

        return n_channels_per_branch_in, n_channels_base_branch_out, n_channels_out


    def __get_n_channels_for_ADA(self, expansion_factor_ADA, n_branches_ADA, n_channels_in_ADA):

        n_channels_out_sep_per_branch_in_out = int(n_channels_in_ADA * expansion_factor_ADA/ float(n_branches_ADA))
        n_channels_out = int(n_channels_out_sep_per_branch_in_out * n_branches_ADA)

        return n_channels_out_sep_per_branch_in_out, n_channels_out


    def __attention_module(self, tensor):

        Num_timesteps = tensor.size()[2]
        Spatial_size = tensor.size()[3]
        a = self.Coeff_Temporal_Attention
        
        if self.Temporal_Attention_Method == 'Linear':
            timeArray = torch.arange(Num_timesteps).to(device='cuda', dtype=torch.float)
            timeArray = a + (1 - a) * (timeArray) // Num_timesteps
            new_tensor = tensor * timeArray.view(1, 1, Num_timesteps, 1, 1)
        else:
            new_tensor = tensor

        if self.Temporal_Attention_Method == 'Gaussian':
            spatialArray = torch.from_numpy(signal.gaussian(Spatial_size, std = self.Coeff_Spatial_Attention)).to(device='cuda', dtype=torch.float)
            new_tensor = new_tensor * spatialArray.view(1, 1, 1, Spatial_size, 1)
            new_tensor = new_tensor * spatialArray.view(1, 1, 1, 1, Spatial_size)

        return  new_tensor

