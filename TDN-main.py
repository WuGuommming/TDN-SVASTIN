import os
import os.path as osp
import time
import warnings

import numpy as np
from importlib_metadata.compat.py39 import ep_matches
from torch.autograd import Variable

from model.model import *
from torch_dwt.functional import idwt3, dwt3

import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from ops.logger import setup_logger
from ops.lr_scheduler import get_scheduler
from ops.utils import reduce_tensor
from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter, accuracy
from tensorboardX import SummaryWriter
# from torch.utils.data import *
import torch.utils.data
import torchvision
import numpy as np
import torch.autograd
from torch import nn


args = parser.parse_args()
torch.manual_seed(324)

zeroinput = 0
oneinput = 1

def main():
    device = torch.device('cuda')

    if args.dataset == "hmdb51":
        args.tune_from = "C:\\data\\TDN_pth\\best-hmdb51\\kin-101.tar"
    elif args.dataset == "ucf101":
        args.tune_from = "C:\\data\\TDN_pth\\best-ucf101\\kin-101.tar"
    else:
        print("dataset error")
        return

    num_class, args.train_list, args.val_list, args.target_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                                      args.modality,
                                                                                                      args.root_dataset)
    full_arch_name = args.arch
    args.store_name = '_'.join(
        [args.store_name, 'TDN_', args.dataset, args.modality, full_arch_name,
         args.consensus_type, 'segment%d' % args.num_segments, 'e{}'.format(args.epochs)])
    if args.pretrain != 'imagenet':
        args.store_name += '_{}'.format(args.pretrain)
    if args.dense_sample:
        args.store_name += '_dense'
    if args.suffix is not None:
        args.store_name += '_{}'.format(args.suffix)

    check_rootfolders()

    logger = setup_logger(output=os.path.join(args.root_model, args.store_name, "log"),
                          distributed_rank=0,
                          name=f'TDN')
    logger.info('storing name: ' + args.store_name)

    model = TSN(num_class,
                args.num_segments,
                args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn,
                pretrain=args.pretrain,
                fc_lr5=(args.tune_from and args.dataset in args.tune_from))

    global zeroinput, oneinput
    zeroinput = torch.zeros(3, 224, 224).cuda()
    oneinput = torch.ones(3, 224, 224).cuda()
    for z, o, m, s in zip(zeroinput, oneinput, model.input_mean, model.input_std):
        z.sub_(m).div_(s)
        o.sub_(m).div_(s)
    zeroinput = zeroinput.repeat(args.batch_size, 40, 1, 1)
    oneinput = oneinput.repeat(args.batch_size, 40, 1, 1)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    for group in policies:
        logger.info(
            ('[TDN-{}]group: {} has {} params, lr_mult: {}, decay_mult: {}'.
             format(args.arch, group['name'], len(group['params']),
                    group['lr_mult'], group['decay_mult'])))

    train_augmentation = model.get_augmentation(
        flip=False if 'something' in args.dataset else True)

    cudnn.benchmark = True

    # Data loading code
    normalize = GroupNormalize(input_mean, input_std)

    train_dataset = TSNDataSet(
        args.dataset,
        args.root_path,
        args.train_list,
        num_segments=args.num_segments,
        modality=args.modality,
        image_tmpl=prefix,
        transform=torchvision.transforms.Compose([train_augmentation,
                                                  Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                                                  ToTorchFormatTensor(
                                                      div=(args.arch not in ['BNInception', 'InceptionV3'])),
                                                  normalize, ]),
        dense_sample=args.dense_sample)

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # train_sampler = torch.utils.data.SequentialSampler(train_dataset)
    # for i in train_sampler:
    #    print(i, end=',')
    # print("\n")
    # train_sampler = torch.utils.data.RandomSampler(train_dataset)
    # for i in train_sampler:
    #    print(i, end=',')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size, num_workers=args.workers,
                                               pin_memory=True, shuffle=True, drop_last=True)

    val_dataset = TSNDataSet(
        args.dataset,
        args.root_path,
        args.val_list,
        num_segments=args.num_segments,
        modality=args.modality,
        image_tmpl=prefix,
        random_shift=False,
        transform=torchvision.transforms.Compose([
            GroupScale(int(scale_size)), GroupCenterCrop(crop_size),
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize, ]),
        dense_sample=args.dense_sample,
        new_length=args.new_length)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size, num_workers=args.workers,
                                             pin_memory=True, sampler=None, shuffle=False, drop_last=True)

    target_dataset = TSNDataSet(
        args.dataset,
        args.root_path,
        args.target_list,
        num_segments=args.num_segments,
        modality=args.modality,
        image_tmpl=prefix,
        random_shift=False,
        transform=torchvision.transforms.Compose([
            GroupScale(int(scale_size)), GroupCenterCrop(crop_size),
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize, ]),
        dense_sample=args.dense_sample,
        new_length=args.new_length)

    target_loader = torch.utils.data.DataLoader(target_dataset,
                                             batch_size=args.batch_size, num_workers=args.workers,
                                             pin_memory=True, sampler=None, shuffle=False, drop_last=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    print(args.tune_from)
    if args.tune_from:
        logger.info(("=> fine-tuning from '{}'".format(args.tune_from)))
        sd = torch.load(args.tune_from)
        sd = sd['state_dict']
        model_dict = model.state_dict()
        replace_dict = []
        if (args.dataset == 'hmdb51' or args.dataset == 'ucf101') and (
                'v1' in args.tune_from or 'v2' in args.tune_from):
            sd = {k.replace('module.base_model.', 'base_model.'): v for k, v in sd.items()}
            sd = {k.replace('module.new_fc.', 'new_fc.'): v for k, v in sd.items()}
        for k, v in sd.items():
            if k not in model_dict and k.replace('.net', '') in model_dict:
                logger.info('=> Load after remove .net: ', k)
                replace_dict.append((k, k.replace('.net', '')))
        for k, v in model_dict.items():
            if k not in sd and k.replace('.net', '') in sd:
                logger.info('=> Load after adding .net: ', k)
                replace_dict.append((k.replace('.net', ''), k))

        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)
        keys1 = set(list(sd.keys()))
        keys2 = set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        logger.info('#### Notice: keys loaded but not in models: {}'.format(keys1 - keys2))
        logger.info('#### Notice: keys required but not in pre-models: {}'.format(keys2 - keys1))
        if args.dataset not in args.tune_from:  # new dataset
            logger.info('=> New dataset, do not load fc weights')
            sd = {k: v for k, v in sd.items() if 'fc' not in k}
        model_dict.update(sd)
        model.load_state_dict(model_dict)

    for p in model.parameters():
        if p.requires_grad:
            p.requires_grad = False

    with open(os.path.join(args.root_model, args.store_name, "log", 'args.txt'), 'w') as f:
        f.write(str(args))

    INN_net = Model().to(device)
    init_model(INN_net)

    total_num = sum(p.numel() for p in INN_net.parameters())
    trainable_num = sum(p.numel() for p in INN_net.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})

    params_trainable = (list(filter(lambda p: p.requires_grad, INN_net.parameters())))
    optim1 = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
    weight_scheduler = torch.optim.lr_scheduler.StepLR(optim1, c.weight_step, gamma=c.gamma)

    '''    
    if args.models == "MVIT":
        channel = 8
        width = 112
    elif args.models == "SLOWFAST":
        channel = 16
        width = 128
    else:'''
    channel = 20
    width = 112

    totalTime = time.time()
    success_number = 0
    number = 0

    model = model.cuda()
    model.eval()

    target_data = {}
    for j, (input_2, target_2) in enumerate(target_loader):
        input_2 = input_2.cuda()
        target_2 = target_2.cuda()

        result_2 = model(input_2)
        labels_2 = torch.argmax(result_2, dim=1)

        if labels_2[0].item() == target_2[0].item():
            target_data[labels_2[0].item()] = (input_2.cpu(), target_2.cpu())

    for i in range(num_class):
        if i not in target_data.keys():
            print(f"error target loader, {i} not exist")
            return

    for i, (input, target) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()

        cover = input
        result = model(input)
        labels = torch.argmax(result, dim=1)

        X_tgt = target_data[labels[0].item()][0].cuda()

        X_1 = torch.full(input.shape, 0.00000000001).to(device)
        # 均匀输入
        X_ori = X_1.to(device)
        X_ori = Variable(X_ori, requires_grad=True)
        optim2 = torch.optim.Adam([X_ori], lr=c.lr2)

        cover_dwt_1 = dwt3(cover.unsqueeze(dim=0).view(args.batch_size, 3, 40, 224, 224), "haar")
        # [1, 8, 3, 20, 112, 112]

        cover_low_0 = cover_dwt_1[0][0]
        # [3, 20, 112, 112]

        cover_dwt_1 = torch.flatten(cover_dwt_1, start_dim=1, end_dim=2)
        # [1, 24, 20, 112, 112]
        for i_epoch in range(c.epochs):
            #################
            #    train:可逆网络的训练过程   #
            #################
            CGT = X_ori.to(device).unsqueeze(dim=0).view(args.batch_size, 3, 40, 224, 224)
            CGT_dwt_1 = dwt3(CGT, "haar")
            CGT_dwt_1 = torch.flatten(CGT_dwt_1, start_dim=1, end_dim=2)
            input_dwt_1 = torch.cat((cover_dwt_1, CGT_dwt_1), 1).to(device)
            output_dwt_1 = INN_net(input_dwt_1).to(device)
            # [1, 48, 20, 112, 112]

            output_steg_dwt_2 = output_dwt_1.narrow(1, 0, 24).to(device)
            output_steg_dwt_2 = output_steg_dwt_2.view(1, 8, 3, channel, width, width)
            # [1, 8, 3, 20, 112, 112]

            output_steg_low_0 = output_steg_dwt_2[0][0]

            output_steg_1 = idwt3(output_steg_dwt_2, 'haar').view(args.batch_size, -1, 224, 224)
            cover = cover.view(args.batch_size, -1, 224, 224)

            output_steg_1 = torch.clamp(output_steg_1, min=zeroinput, max=oneinput).to(device)
            eta = torch.clamp(output_steg_1 - cover, min=-c.eps, max=c.eps)
            output_steg_1 = torch.clamp(cover + eta, min=zeroinput, max=oneinput)

            data = output_steg_1
            result = model(data)

            adv_cost = nn.CrossEntropyLoss().to(device)
            MSE = torch.nn.MSELoss(reduction='mean')
            adv_loss = - c.lambda_a * adv_cost(result, target).to(device)
            loss_2 = c.beta_a * torch.mean(torch.sqrt(torch.sum(torch.pow(eta.view(args.batch_size * 3 * 40, 224 * 224), 2), dim=1)))
            ll_loss = c.gama_a * MSE(cover_low_0, output_steg_low_0)
            total_loss = adv_loss + loss_2 + ll_loss
            optim1.zero_grad()
            optim2.zero_grad()

            total_loss.backward()
            optim1.step()

            data = CGT
            C_result = model(data)
            C_adv_loss = - adv_cost(C_result, target).to(device)
            MSE = torch.nn.MSELoss(reduction='mean')
            loss_mse = MSE(CGT.view(-1, 120, 224, 224), X_tgt).to(device)
            total_loss_tgt = c.lambda_b * loss_mse + c.beta_b * C_adv_loss
            total_loss_tgt.backward()
            optim2.step()

            weight_scheduler.step()
            lr_min = c.lr_min
            lr_now = optim1.param_groups[0]['lr']
            if lr_now < lr_min:
                optim1.param_groups[0]['lr'] = lr_min
            if i_epoch % 100 == 0:
                print("Train epoch = {}; adv_loss = {:.5f}; loss_2 ={:.3f}; ll_loss={:.3f}; total_loss = {:.3f}; pre_label = {}; adv_label = {}."
                .format(
                        i_epoch, adv_loss, loss_2.item(), ll_loss.item(), total_loss.item(),
                        target[0].item(),
                        torch.argmax(result, dim=1).item()))

            if not torch.argmax(result, dim=1).item() == target[0].item():
                success_number = success_number + 1
                print(
                    "Success train epoch = {}; adv_loss = {:.5f}; loss_2 ={:.3f}; ll_loss={:.3f}; total_loss = {:.3f}; pre_label = {}; adv_label = {}."
                    .format(
                        i_epoch, adv_loss, loss_2.item(), ll_loss.item(), total_loss.item(),
                        target[0].item(),
                        torch.argmax(result, dim=1).item()))
                break
            else:
                if i_epoch == c.epochs - 1:
                    break
        print(f"{i}/{len(train_loader)}" + "=" * 80)
    print("success_number = ", success_number)
    totalstop_time = time.time()
    time_cost = totalstop_time - totalTime
    print("Total cost time :" + str(time_cost))

    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()

        cover = input
        result = model(input)
        labels = torch.argmax(result, dim=1)

        X_tgt = target_data[labels[0].item()][0].cuda()

        X_1 = torch.full(input.shape, 0.00000000001).to(device)
        # 均匀输入
        X_ori = X_1.to(device)
        X_ori = Variable(X_ori, requires_grad=True)
        optim2 = torch.optim.Adam([X_ori], lr=c.lr2)

        cover_dwt_1 = dwt3(cover.unsqueeze(dim=0).view(args.batch_size, 3, 40, 224, 224), "haar")
        # [1, 8, 3, 20, 112, 112]

        cover_low_0 = cover_dwt_1[0][0]
        # [3, 20, 112, 112]

        cover_dwt_1 = torch.flatten(cover_dwt_1, start_dim=1, end_dim=2)
        # [1, 24, 20, 112, 112]
        for i_epoch in range(c.epochs):
            CGT = X_ori.to(device).unsqueeze(dim=0).view(args.batch_size, 3, 40, 224, 224)
            CGT_dwt_1 = dwt3(CGT, "haar")
            CGT_dwt_1 = torch.flatten(CGT_dwt_1, start_dim=1, end_dim=2)
            input_dwt_1 = torch.cat((cover_dwt_1, CGT_dwt_1), 1).to(device)
            output_dwt_1 = INN_net(input_dwt_1).to(device)
            # [1, 48, 20, 112, 112]

            output_steg_dwt_2 = output_dwt_1.narrow(1, 0, 24).to(device)
            output_steg_dwt_2 = output_steg_dwt_2.view(1, 8, 3, channel, width, width)
            # [1, 8, 3, 20, 112, 112]

            output_steg_low_0 = output_steg_dwt_2[0][0]

            output_steg_1 = idwt3(output_steg_dwt_2, 'haar').view(args.batch_size, -1, 224, 224)
            cover = cover.view(args.batch_size, -1, 224, 224)

            output_steg_1 = torch.clamp(output_steg_1, min=zeroinput, max=oneinput).to(device)
            eta = torch.clamp(output_steg_1 - cover, min=-c.eps, max=c.eps)
            output_steg_1 = torch.clamp(cover + eta, min=zeroinput, max=oneinput)

            data = output_steg_1
            result = model(data)

            adv_cost = nn.CrossEntropyLoss().to(device)
            MSE = torch.nn.MSELoss(reduction='mean')
            adv_loss = - c.lambda_a * adv_cost(result, target).to(device)
            loss_2 = c.beta_a * torch.mean(
                torch.sqrt(torch.sum(torch.pow(eta.view(args.batch_size * 3 * 40, 224 * 224), 2), dim=1)))
            ll_loss = c.gama_a * MSE(cover_low_0, output_steg_low_0)
            total_loss = adv_loss + loss_2 + ll_loss

            optim2.zero_grad()
            data = CGT
            C_result = model(data)
            C_adv_loss = - adv_cost(C_result, target).to(device)
            MSE = torch.nn.MSELoss(reduction='mean')
            loss_mse = MSE(CGT.view(-1, 120, 224, 224), X_tgt).to(device)
            total_loss_tgt = c.lambda_b * loss_mse + c.beta_b * C_adv_loss
            total_loss_tgt.backward()
            optim2.step()

            weight_scheduler.step()
            lr_min = c.lr_min
            lr_now = optim1.param_groups[0]['lr']
            if lr_now < lr_min:
                optim1.param_groups[0]['lr'] = lr_min
            if i_epoch % 100 == 0:
                print(
                    "Valid epoch = {}; adv_loss = {:.5f}; loss_2 ={:.3f}; ll_loss={:.3f}; total_loss = {:.3f}; pre_label = {}; adv_label = {}."
                    .format(
                        i_epoch, adv_loss, loss_2.item(), ll_loss.item(), total_loss.item(),
                        target[0].item(),
                        torch.argmax(result, dim=1).item()))

            if not torch.argmax(result, dim=1).item() == target[0].item():
                success_number = success_number + 1
                print(
                    "Success valid epoch = {}; adv_loss = {:.5f}; loss_2 ={:.3f}; ll_loss={:.3f}; total_loss = {:.3f}; pre_label = {}; adv_label = {}."
                    .format(
                        i_epoch, adv_loss, loss_2.item(), ll_loss.item(), total_loss.item(),
                        target[0].item(),
                        torch.argmax(result, dim=1).item()))
                break
            else:
                if i_epoch == c.epochs - 1:
                    break
        print(f"{i}/{len(val_loader)}" + "=" * 80)
    print("valid success_number = ", success_number)

def save_checkpoint(state, epoch, is_best):
    filename = '%s/%s/%d_epoch_ckpt.pth.tar' % (args.root_model, args.store_name, epoch)
    torch.save(state, filename)
    if is_best:
        best_filename = '%s/%s/best.pth.tar' % (args.root_model, args.store_name)
        torch.save(state, best_filename)


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [
        args.root_log, args.root_model,
        os.path.join(args.root_log, args.store_name),
        os.path.join(args.root_model, args.store_name)
    ]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.makedirs(folder)


main()
