import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import sys
import time
import logging
import datetime
from pathlib import Path
sys.path.append('/home/mahdiar/Projects/AQA_head')
from AQA_head.configs import load_config
from I3D.pytorch_i3d import InceptionI3d

import torch
import torch.utils.data

from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


from AQA_head.core import utils, pytorch_utils, image_utils, metrics
from AQA_head.experiments.AQA_Full_Model import Model_head
from I3D.AQA_dataset import AQA_dataset as Dataset
import I3D.videotransforms as videotransforms

logger = logging.getLogger(__name__)


def train_tco(cfg):
    # get some configs for the training
    n_epochs = cfg.TRAIN.Num_Epochs
    dataset_name = cfg.Dataset.Name
    model_name = '%s_%s' % (cfg.MODEL.Name, utils.timestamp())
    device = 'cuda'

    # data generators
    loader_tr = __define_loader(is_training=True, cfg=cfg)
    loader_te = __define_loader(is_training=False, cfg=cfg)

    logger.info('--- start time')
    logger.info(datetime.datetime.now())

    # load model_i3d
    n_batches_tr = int(cfg.TRAIN.Num_Videos/cfg.TRAIN.Batch_Size)
    n_batches_te = int(cfg.TEST.Num_Videos/cfg.TEST.Batch_Size)

    if cfg.MODEL.Pretrain == 'Kinetics':
        i3d = InceptionI3d(400, in_channels=3)
        i3d.replace_logits(400)
    else:
        i3d = InceptionI3d(157, in_channels=3)
        i3d.replace_logits(157)


    load_model = cfg.Dataset.Pretrained_Charades_Root
    i3d.load_state_dict(torch.load(load_model))
    model_i3d = i3d.cuda()

    # load model_head
    model_head, optimizer, loss_fn, metric_fn, metric_fn_name = __define_AQA_model(device, cfg=cfg)
    decayRate = cfg.SOLVER.Learning_Rate_Decay
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

    # save the model
    model_saver = pytorch_utils.ModelSaver(model_head, dataset_name, model_name)
    model_head.load_state_dict(torch.load('/home/mahdiar/Projects/pytorch-i3d/models/imagenet_adam_0025_Snowboarding/231.pt'))

    # loop on the epochs
    sys.stdout.write('\n')
    for idx_epoch in range(n_epochs):

        epoch_num = idx_epoch + 1

        loss_tr = 0.0
        loss_te = 0.0

        tt1 = time.time()

        y_true_all_train = []
        y_pred_all_train = []
        y_true_all_test = []
        y_pred_all_test = []


        # flag model as training
        model_head.train()
        model_i3d.eval()


        # training
        for idx_batch, (x, heatmap_info, y_true) in enumerate(loader_tr):

            batch_num = idx_batch + 1
            x, y_true = x.to(device), y_true.to(device)
            c = model_i3d(x)

            x_head_1 = c.to(device)

            optimizer.zero_grad()

            y_pred = model_head((x_head_1, heatmap_info.cuda()))

            loss = loss_fn(y_pred.squeeze(1), y_true)
            if cfg.MODEL.Loss_Type == 'Mixed':
                KL_loss = torch.distributions.kl_divergence(torch.distributions.Normal(y_true.mean(), y_true.std()),
                                                            torch.distributions.Normal(y_pred.mean(), y_pred.std())).mean()

                loss =  loss + cfg.SOLVER.KL_coeff * KL_loss
            loss.backward()

            optimizer.step()
            # my_lr_scheduler.step()

            # calculate accuracy
            y_true = y_true.cpu().numpy()
            y_pred = y_pred.cpu().detach().numpy()
            y_true_all_train.extend(y_true)
            y_pred_all_train.extend(y_pred.squeeze(1))

            loss_b_tr = loss.cpu().detach().numpy()
            # metric_fn_val = metric_fn(y_true, y_pred)

            loss_tr += float(loss_b_tr)
            loss_b_tr = loss_tr / float(batch_num)
            tt2 = time.time()
            duration = tt2 - tt1
            sys.stdout.write('\r%04ds - epoch: %04d/%04d, batch [tr]: %04d/%04d, loss : %0.4f ' % (
            duration, epoch_num, n_epochs, batch_num, n_batches_tr, loss_b_tr))

        # flag model as testing

        model_head.eval()

        # testing
        for idx_batch, (x, heatmap_info, y_true) in enumerate(loader_te):

            batch_num = idx_batch + 1
            x, y_true = x.to(device), y_true.to(device)
            c = model_i3d(x)

            x_head_1 = c.to(device)

            x_head = (x_head_1, heatmap_info.cuda())

            y_pred = model_head(x_head)

            loss_b_te = loss_fn(y_pred.squeeze(1), y_true).cpu().detach().numpy()
            y_true = y_true.cpu().numpy()
            y_pred = y_pred.cpu().detach().numpy()

            y_true_all_test.extend(y_true)
            y_pred_all_test.extend(y_pred.squeeze(1))

            loss_te += float(loss_b_te)
            loss_b_te = loss_te / float(batch_num)
            tt2 = time.time()
            duration = tt2 - tt1
            sys.stdout.write('\r%04ds - epoch: %04d/%04d, batch [te]: %04d/%04d, loss: %0.4f' % (
            duration, epoch_num, n_epochs, batch_num, n_batches_te, loss_b_te))

        loss_tr /= float(n_batches_tr)
        loss_te /= float(n_batches_te)
        metric_fn_value_train = metric_fn(y_pred_all_train, y_true_all_train)
        metric_fn_value_test = metric_fn(y_pred_all_test, y_true_all_test)


        tt2 = time.time()
        duration = tt2 - tt1
        sys.stdout.write('\r%04ds - epoch: %04d/%04d, [tr]: loss: %0.4f, Sp.Corr.: %0.4f, [te]: loss: %0.4f, Sp.Corr.%0.4f           \n' % (
        duration, epoch_num, n_epochs, loss_tr, metric_fn_value_train[0], loss_te, metric_fn_value_test[0]))

        # after each epoch, save data
        if idx_epoch%25==0:
            model_saver.save(idx_epoch)

    logger.info('--- finish time')
    logger.info(datetime.datetime.now())


def __define_loader(is_training, cfg):
    """
    Define data loader.
    """

    root_AQA_dataset = ('/home/mahdiar/Projects/pytorch-i3d/AQA_Dataset/%s'% cfg.Dataset.Sport_Field)
    train_split_file = ('/home/mahdiar/Projects/pytorch-i3d/AQA_Dataset/split_4_train_list.mat')
    val_split_file = ('/home/mahdiar/Projects/pytorch-i3d/AQA_Dataset/split_4_test_list.mat')
    mode = 'rgb'

    # get some configs for the training
    n_classes = cfg.Dataset.Num_Classes
    dataset_name = cfg.Dataset.Name
    n_timesteps = cfg.MODEL.Num_Timesteps
    n_workers = cfg.TRAIN.Num_Workers

    batch_size_tr = cfg.TRAIN.Batch_Size
    batch_size_te = cfg.TEST.Batch_Size
    batch_size = batch_size_tr if is_training else batch_size_te

    if cfg.MODEL.BackBone == 'i3d_rgb':
        feature_name = 'Mixed_5c'
        c = 1024
        h = 7
        w = 7
    else:
        raise KeyError
    feature_dim = (c, n_timesteps, h, w)

    # data generators

    train_transforms = transforms.Compose([videotransforms.CenterCrop(224),
                                           videotransforms.RandomHorizontalFlip(),
                                           ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(split_file=train_split_file, root=root_AQA_dataset, mode=mode, cfg=cfg, transforms=train_transforms, Train_or_Test=1)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)

    val_dataset = Dataset(split_file=val_split_file, root=root_AQA_dataset, mode=mode, cfg=cfg, transforms=test_transforms, Train_or_Test=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)

    if is_training:
        data_loader = dataloader
    else:
        data_loader = val_dataloader

    return data_loader


def __define_AQA_model(device, cfg):
    """
    Define model, optimizer, loss function and metric function.
    """
    # some configurations
    classification_type = cfg.MODEL.Classification_Type
    solver_name = cfg.SOLVER.Name
    solver_lr = cfg.SOLVER.Learning_Rate
    adam_epsilon = cfg.SOLVER.ADAM_Epsilon
    adam_w_decay = cfg.SOLVER.ADAM_Weight_Decay
    # define model
    model = Model_head()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count())).to(device)
    model_param = model.parameters()

    # define the optimizer
    optimizer = SGD(model_param, lr=0.00005) if solver_name == 'sgd' else Adam(model_param, lr=solver_lr, eps=adam_epsilon)

    # loss and evaluation function for either multi-label "ml" or single-label "sl" classification
    if classification_type == 'ml':
        loss_fn = torch.nn.BCELoss()
        metric_fn = metrics.map_charades
        metric_fn_name = 'map'
    else:
        loss_fn = torch.nn.MSELoss()
        metric_fn = metrics.spearmanCorr
        metric_fn_name = 'Sp.Corr.'

    return model, optimizer, loss_fn, metric_fn, metric_fn_name


def __main():
    """
    Run this script to train AQA.
    """

    # Dataset = 'Charades'
    Dataset = 'AQA'

    path = Path(os.getcwd())

    if Dataset == 'Charades':
        config_path = '%s/AQA_head/configs/configurations_charades.yml' % path.parent
    else:
        config_path = '%s/AQA_head/configs/configurations.yml' % path

    cfg = load_config.load_config(config_path)

    train_tco(cfg)


if __name__ == '__main__':
    __main()
