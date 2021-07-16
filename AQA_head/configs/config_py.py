import yaml

document = {

'Num_GPUS': 2,
'LOG_num': 10,
'Dataset': {
  'Name':'AQA',
  'Num_Classes': 1,
  'Sport_Field': 'Diving',
  'Index_Field': 1,
  'Pretrained_Charades_Root' : '/home/mahdiar/Projects/pytorch-i3d/models/rgb_imagenet.pt',
  'Num_Frames': 103
},

'MODEL':{
  'Name': 'AQA',
  'Loss_Type': 'MSE',
  'Num_AQA_Layers': 3,
  'Num_JCA_Layers': 2,
  'Num_ADA_Layers': 2,
  'Num_Branches_JCA': 3,
  'Num_Branches_ADA': 3,
  'Expansion_Factor_JCA': 13/12,
  'Expansion_Factor_ADA': 1.5,
  'BackBone': 'i3d_rgb',
  'BackBone_FeatureLayer': 4,
  'Num_Timesteps': int(103/8),
  'Spatial_transform_size': 224,
  'Pose_Model': 'HRNet',
  'Pose_Pretrained_Root' : '/home/mahdiar/Projects/pytorch-i3d/Pose/models/coco_pose_iter_440000.pth.tar',
  'Classification_Type': 'sl',
  'Pretrain': 'Kinetics',
  'Ablated': 'N',
  'Pose_Spatial_Size': 64,
  'JCA_Expansion_Balance': 10,
  'Spatial_Attention_Method': 'Gaussian',
  'Temporal_Attention_Method': 'N',
  'Coeff_Spatial_Attention': 5,
  'Coeff_Temporal_Attention': 0.9,

},

'TRAIN':{
  'Batch_Size': 20,
  'Num_Epochs': 1000,
  'Num_Workers': 4,
  'Num_Videos' : 300,
},

'TEST':{
  'Batch_Size': 14,
  'Num_Videos' : 28,
},

'SOLVER':{
  'Name': 'adam',
  'Learning_Rate': 0.005,
  'Learning_Rate_Decay': 1,
  'ADAM_Epsilon': 0.0001,
  'SGD_Weight_Decay': 0.0001,
  'SGD_Momentum': 0.9,
  'SGD_Nesterov': True,
  'ADAM_Weight_Decay': 0.000001,

}
}

with open('configurations.yml', 'w') as outfile:
    yaml.dump(document, outfile, default_flow_style=False)
