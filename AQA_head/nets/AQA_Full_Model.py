
from torch.nn import Module, Dropout, BatchNorm1d, LeakyReLU, Linear, LogSoftmax, Sigmoid, PReLU
from AQA_head.configs import load_config
from AQA_head.core import utils
import os
from pathlib import Path
from AQA_head.nets import AQA
import torch

class Model_head(Module):

    """
    Define Action Quality Assessor.
    """
    def __init__(self):
        super(Model_head, self).__init__()

        # some configurations for the model

        Dataset = 'Charades'
        path = Path(os.getcwd())

        cfg = load_config.load_config('%s/AQA_head/configs/configurations.yml' % path)

        Num_AQA_Layers = cfg.MODEL.Num_AQA_Layers
        Num_JCA_Layers = cfg.MODEL.Num_JCA_Layers
        Num_ADA_Layers = cfg.MODEL.Num_ADA_Layers
        Expansion_Factor_JCA = cfg.MODEL.Expansion_Factor_JCA
        Expansion_Factor_ADA = cfg.MODEL.Expansion_Factor_ADA
        Num_Branches_JCA = cfg.MODEL.Num_Branches_JCA
        Num_Branches_ADA = cfg.MODEL.Num_Branches_ADA
        BackBone = cfg.MODEL.BackBone
        BackBone_FeatureLayer = cfg.MODEL.BackBone_FeatureLayer
        Num_Timesteps = cfg.MODEL.Num_Timesteps
        Pose_Spatial_Size = cfg.MODEL.Pose_Spatial_Size
        n_classes = cfg.Dataset.Num_Classes
        figure_skating_flag = 0
        Ablation_flag = cfg.MODEL.Ablated
        Spatial_Attention_Method = cfg.MODEL.Spatial_Attention_Method
        Temporal_Attention_Method = cfg.MODEL.Spatial_Attention_Method
        Coeff_Spatial_Attention = cfg.MODEL.Coeff_Spatial_Attention
        Coeff_Temporal_Attention = cfg.MODEL.Coeff_Temporal_Attention

        Pose_Model = cfg.MODEL.Pose_Model
        if Pose_Model == 'Openpose':
            Num_Joints = 19
        elif Pose_Model == 'HRNet':
            Num_Joints = 16
        elif Pose_Model == 'Openpose_Figure':
            Num_Joints = 25
            figure_skating_flag = 1
        else:
            raise ValueError ('Other pose estimation tools not are not implemented yet')

        if cfg.Dataset.Sport_Field != 'Figure_skating':
            c = 1024
            h = 7
            w = 7
            input_shape_JCA = (None, Num_Joints, cfg.Dataset.Num_Frames, Pose_Spatial_Size, Pose_Spatial_Size)
        else:
            c = 4096
            h = 1
            w = 1
            input_shape_JCA = (None, Num_Joints, int(cfg.Dataset.Num_Frames/2), Pose_Spatial_Size, Pose_Spatial_Size)

        input_shape_ADA = (None,c,Num_Timesteps,h,w)

        self.AQA = AQA.AQA(input_shape_JCA = input_shape_JCA, input_shape_ADA=input_shape_ADA,
                               n_layers_JCA = Num_JCA_Layers, n_layers_ADA = Num_ADA_Layers,
                               n_branches_JCA = Num_Branches_JCA, n_branches_ADA = Num_Branches_ADA,
                               expansion_factor_ADA= Expansion_Factor_ADA,
                               expansion_factor_JCA=Expansion_Factor_JCA,
                               final_expansion_JCA_for_balance= cfg.MODEL.JCA_Expansion_Balance,
                               figure_skating_flag= figure_skating_flag, Ablated_flag= Ablation_flag,
                               Spatial_Attention_Method = Spatial_Attention_Method, Temporal_Attention_Method = Temporal_Attention_Method,
                               Coeff_Spatial_Attention = Coeff_Spatial_Attention, Coeff_Temporal_Attention = Coeff_Temporal_Attention
                               )


        # get number of output channels after timeception
        n_channels_in = self.AQA.n_channels_out

        # define layers for classifier
        # self.do1 = Dropout(0.5)

        self.do1 = Dropout(0.5)
        self.l1 = Linear(n_channels_in, 512)
        self.bn1 = BatchNorm1d(512)
        self.ac1 = LeakyReLU(0.2)
        self.do2 = Dropout(0.25)
        self.l2 = Linear(512, n_classes)


    def forward(self, input):
        # feedforward the input to the timeception layers
        tensor = self.AQA(input)

        # max-pool over space-time
        bn, c, t, h, w = tensor.size()
        tensor = tensor.view(bn, c, t * h * w)
        tensor = torch.max(tensor, dim=2, keepdim=False)
        tensor = tensor[0]

        # dense layers for classification
        tensor = self.do1(tensor)
        tensor = self.l1(tensor)
        tensor = self.bn1(tensor)
        tensor = self.ac1(tensor)
        tensor = self.do2(tensor)
        tensor = self.l2(tensor)

        return tensor
