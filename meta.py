import argparse
import os
import yaml
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--name', default='paperset') 
parser.add_argument('--phase', type=str, default='meta', choices=['pretrain', 'meta'])
parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet', 'CIFAR-FS','FG-CUB','FG-Dogs','FG-Cars']) 
parser.add_argument('--n_shot', type=int, default=5, choices=[1,5])
parser.add_argument('--num_workers', type=int, default=8, choices=[16,8,4,2,1])
parser.add_argument('--image_size', type=int,default=224) 
parser.add_argument('--ef_epoch', type=int,default=1)
parser.add_argument('--episode', type=int,default=600)
parser.add_argument('--text_length', type=int,default=20)
parser.add_argument('--eqnorm', action='store_false')
parser.add_argument('--aug_support', type=int, default=1)
parser.add_argument('--backbone', type=str, default='visformer-t')
parser.add_argument('--load_dir', type=str, default='')
parser.add_argument('--load_local_dir', type=str, default='')
parser.add_argument('--lr', type=float, default=1e-6)
parser.add_argument('--projector_lr', type=float, default=5e-4)
parser.add_argument('--epoch', type=int,default=100)
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--stage', type=float, default=3.2, choices=[2, 2.1, 2.2, 2.3, 3, 3.1, 3.2, 3.3])
parser.add_argument('--t', type=float, default=0.2)
parser.add_argument('--avg', type=str, default='all', choices=['all', 'patch', 'head'])
parser.add_argument('--gpu', default='2')
args = parser.parse_args() 
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import random
from tqdm import tqdm
from timm.scheduler import CosineLRScheduler
from timm.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import scipy.stats

import clip
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

from model import visformer
from datas.datasets_meta import Datasets
from datas.dataloader import EpisodeSampler
import utils


def fix_random_seeds(seed=12345):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_text_feature(teacher, dataset,args):
        class_idx = dataset.dataset.classes
        idx2text = dataset.idx2text
        text = ['A photo of ' + idx2text[idx] for idx in class_idx]

        teacher.eval()
        text_token = clip.tokenize(text).cuda()
        if args.text_length != -1:
            text_token = text_token[:, :args.text_length]

        with torch.no_grad():
            text_feature = teacher.encode_text(text_token)
            text_feature = text_feature.float()

        return text_feature


class SoftTargetCrossEntropy(torch.nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        N_rep = x.shape[0]
        N = target.shape[0]
        if not N==N_rep:
            target = target.repeat(N_rep//N,1)
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


### token labeling revision ###
# 1.topk (k=3 or 5) one-hot label, then use softmax to generate soft target (using soft cross-entropy loss to supervise patch token)
# 2.teacher model use global classifier 
def generate_softlabel(logits, smoothing=0.1, k=3, bp=10, device='cuda'):
    n_classes = logits.size(1)  #64
    off_value = smoothing / n_classes
    on_value = 1 - smoothing + off_value
    logits_max, _ = logits.max(dim=1, keepdim=True) #(5,1,8,7)---(5,1,7,7)
    b, c, h, w = logits_max.size()
    logits_max = logits_max.view(b, c, h*w) #(5,1,56)
    _, pos_select = logits_max.topk(h*w - bp, dim=-1)   #(5,1,46)
    pos_mask = torch.zeros_like(logits_max).scatter(-1, pos_select, 1)  #(5,1,56)
    pos_mask = pos_mask.permute(0, 2, 1).view(-1, 1)
    #pos_select = logits_max.topk(h*w - bp)
    #pos_select = pos_select.permute(0, 2, 3, 1).view(-1, 1)
    logits = logits.permute(0, 2, 3, 1).view(-1, n_classes) #(280,64)
    value, idx = logits.topk(k)
    bg_map = torch.full(pos_mask.size(), c, device=device)

    soft_label = torch.full((logits.size(0), logits.size(1)+1), off_value, device=device).scatter_(1, idx, on_value)    #(280,65)
    soft_bg_label = torch.full((logits.size(0), logits.size(1)+1), off_value, device=device).scatter_(1, bg_map, on_value)
    soft_label = soft_label * pos_mask + soft_bg_label * (1 - pos_mask)
    return soft_label


def main(config):
    fix_random_seeds(args.seed)
    
    svname = args.name+'_{}shot'.format(args.n_shot)
    if svname is None:
        svname = '{}-{}'.format(args.phase, args.backbone)
        '''
        if args.augment=='crop_aug':
            svname += '-' + 'crop'
        if args.repeat_aug:
            svname += '-' + 'repeat'
        '''
    save_path = os.path.join('./save', args.dataset, svname)
    utils.ensure_path(save_path)       
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))  
    utils.log('set gpu: {} '.format(args.gpu))
    utils.log(vars(args))


    # train_loader #
    n_way=5
    n_train_way=5
    n_shot=args.n_shot
    ef_epoch = args.ef_epoch
    episodes=args.episode
    
    args_dict = vars(args)
    train_dataset = Datasets(args.dataset, split='train', **args_dict)
    n_episodes = int(len(train_dataset) / (n_train_way * (n_shot + 15)))
    episode_sampler = EpisodeSampler(train_dataset.dataset.targets,
                                     n_episodes,
                                     n_train_way,
                                     n_shot + 15, fix_seed=False)
    train_loader = DataLoader(train_dataset, batch_sampler=episode_sampler, num_workers=args.num_workers)
    utils.log('train dataset: {} (x{}), {}'.format(train_dataset[0][0].shape, len(train_dataset),len(train_dataset.dataset.classes)))
    utils.visualize_dataset(train_dataset, 'train_dataset', writer)
    num_classes = len(train_dataset.dataset.classes)
    # num_classes=1000
    
    # val_loader #
    val_dataset = Datasets(args.dataset, split='test', **args_dict)
    episode_sampler = EpisodeSampler(val_dataset.dataset.targets, 
                                         episodes, 
                                         n_way, 
                                         n_shot + 15)
    val_loader = DataLoader(val_dataset, batch_sampler=episode_sampler, num_workers=args.num_workers)
    utils.log('val dataset: {} (x{}), {}'.format(val_dataset[0][0].shape, len(val_dataset),len(val_dataset.dataset.classes)))
    utils.visualize_dataset(val_dataset, 'val_dataset', writer)
    
    # text
    device=f"cuda:{0}"
    teacher, _ = clip.load("ViT-B/32",device=device)
    text_dim = 512
    if args.text_length !=-1:
        teacher.context_length = args.text_length
        teacher.positional_embedding.data = teacher.positional_embedding.data[:args.text_length]
        for layer in teacher.transformer.resblocks:
            layer.attn_mask.data = layer.attn_mask.data[:args.text_length, :args.text_length]
    train_text = get_text_feature(teacher, train_dataset,args)
    val_text = get_text_feature(teacher, val_dataset,args)

    # equal length normalization for features
    if args.eqnorm:
        avg_length = (train_text ** 2).sum(-1).sqrt().mean().item()
        train_text = F.normalize(train_text, dim=-1) * avg_length
        val_text = F.normalize(val_text, dim=-1) * avg_length

    # Model #
    if args.backbone == 'visformer-t':
        model = visformer.visformer_tiny(num_classes=num_classes, phase=args.phase)
        model_teacher = visformer.visformer_tiny(num_classes=num_classes, phase=args.phase)
    elif args.backbone == 'visformer-t-84':
        model = visformer.visformer_tiny_84(num_classes=num_classes, phase=args.phase)
        model_teacher = visformer.visformer_tiny_84(num_classes=num_classes, phase=args.phase)
    else:
        raise ValueError(f'unknown model: {args.backbone}')     
    
    # text to image 
    feature_dim = 384
    #stage3
    model.t2i = torch.nn.Linear(text_dim, feature_dim, bias=False)
    model.t2i1 = torch.nn.Linear(text_dim, feature_dim, bias=False)
    model.se_block = torch.nn.Sequential(torch.nn.Linear(feature_dim*2, feature_dim, bias=True),
                                               torch.nn.Sigmoid(),
                                               torch.nn.Linear(feature_dim, feature_dim),
                                               torch.nn.Sigmoid(),)
    #stage2
    model.t2i2 = torch.nn.Linear(text_dim, 192, bias=False)
    model.t2i3 = torch.nn.Linear(text_dim, 192, bias=False)
    model.se_block1 = torch.nn.Sequential(torch.nn.Linear(192*2, 192, bias=True),
                                               torch.nn.Sigmoid(),
                                               torch.nn.Linear(192, 192),
                                               torch.nn.Sigmoid(),)
    
    model.classifier_local=torch.nn.Linear(feature_dim ,num_classes+1)
    model_teacher.classifier_local=torch.nn.Linear(feature_dim ,num_classes+1)

    # load modal
    if args.resume:
        init = args.resume
    else:
        # init ="save/"+args.dataset+'/'+'pretrain'+'/'+'epoch-600.pth'
        init ="save/"+args.dataset+'/'+'pretrain'+'/'+'best-{}shot.pth'.format(args.n_shot)
        checkpoint = torch.load(init,map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        # init_local ="save/"+args.dataset+'/'+'pretrain'+'/'+'epoch-600.pth'
        init_local ="save/"+args.dataset+'/'+'pretrain'+'/'+'best-{}shot.pth'.format(args.n_shot)
        checkpoint_local = torch.load(init_local,map_location=device)
        model.load_state_dict(checkpoint_local['state_dict'], strict=False)

    
    utils.log('num params: {}'.format(utils.compute_n_params(model)))
    model=model.cuda()
    model_teacher=model_teacher.cuda()

    # Optimizer
    optim_params_id = [id(param) for param in model.t2i.parameters()]
    optim_params_id += [id(param) for param in model.t2i1.parameters()]
    optim_params_id += [id(param) for param in model.t2i2.parameters()]
    optim_params_id += [id(param) for param in model.t2i3.parameters()]

    optim_params = [param for param in model.parameters() if id(param) in optim_params_id]
    other_params = [param for param in model.parameters() if id(param) not in optim_params_id]
    
    # low lr of backbone
    optimizer = AdamW([{'params': optim_params, 'lr':args.projector_lr, 'weight_decay': 5e-2},
                        {'params': other_params, 'lr': args.lr}], weight_decay=5e-2)
    lr_scheduler=None
 
    # Other parameters #
    criterion_TL = SoftTargetCrossEntropy()
    max_epoch = args.epoch
    save_epoch = 100
    max_va = 0.
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()
    if args.resume:
        checkpoint = torch.load(args.resume,map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print(f'load checkpoint at epoch {start_epoch}')
    else:
         start_epoch = 1

    
    # Train and Val #
    for epoch in range(start_epoch, max_epoch+1):
        timer_epoch.s()
        aves_keys = ['tl', 'ta', 'vl', 'va']
        aves = {k: utils.Averager() for k in aves_keys}
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
    
        # train
        model.train()
        for episode in tqdm(train_loader, desc='train', leave=False):
            image = episode[0].cuda()  # way * (shot+15)
            glabels = episode[1].cuda()
            labels = torch.arange(n_train_way).unsqueeze(-1).repeat(1, 15).view(-1).cuda()
            
            image = image.view(n_train_way, n_shot+15, *image.shape[1:])
            sup, que = image[:, :n_shot].contiguous(), image[:, n_shot:].contiguous()
            sup, que = sup.view(-1, *sup.shape[2:]), que.view(-1, *que.shape[2:])

            glabels = glabels.view(n_train_way, n_shot+15)[:, :n_shot]
            glabels = glabels.contiguous().view(-1)
            text_features = train_text[glabels]

            _, sup_im_features, logits_token,_, _ = model.forward_with_semantic_prompt(sup, text_features,args)
            sup_im_features=sup_im_features.cuda()

            # local tokens
            with torch.no_grad():
                logits_t, _, _, logits_token_t,token_t = model_teacher(sup,args)
                soft_label = generate_softlabel(logits_token_t, k=5, bp=10) #(280,65)
            # self promoted token labeling 
            b, c, h, w = logits_token_t.size()
            # centerness or sharpen, currently omitted
            logits_flatten = logits_token.permute(0, 2, 3, 1).view(-1, c+1)   #(280,65)
            token_loss = criterion_TL(logits_flatten, soft_label)

            # consistency_loss
            _,sup_im_features_pre,_,_,_ = model(sup,args)
            consistency_loss = F.mse_loss(sup_im_features, sup_im_features_pre)
            
            sup_im_features = sup_im_features.view(n_train_way, n_shot, -1).mean(dim=1)
            _,que_im_features,_,_,_= model(que,args)
            logits = F.normalize(que_im_features, dim=-1) @ F.normalize(sup_im_features, dim=-1).t()
            loss = F.cross_entropy(logits/ args.t, labels)
            acc = utils.compute_acc(logits, labels)

            # total_loss= loss + model.loss_weight * consistency_loss + model.loss_weight1 * token_loss
            total_loss= loss + model.loss_weight * consistency_loss + model.loss_weight1 * token_loss


            optimizer.zero_grad()
            total_loss.backward()
            # loss.backward()
            optimizer.step()

            aves['tl'].add(loss.item())
            aves['ta'].add(acc)

        # eval
        if epoch % ef_epoch == 0 or epoch==1:
            model.eval()
            with torch.no_grad():
                for episode in tqdm(val_loader, desc='val', leave=False):
                    image = episode[0].cuda()  # way * (shot+15)
                    glabels = episode[1].cuda()
                    labels = torch.arange(n_train_way).unsqueeze(-1).repeat(1, 15).view(-1).cuda()
                    
                    image = image.view(n_train_way, n_shot+15, *image.shape[1:])
                    sup, que = image[:, :n_shot].contiguous(), image[:, n_shot:].contiguous()
                    sup, que = sup.view(-1, *sup.shape[2:]), que.view(-1, *que.shape[2:])

                    glabels = glabels.view(n_train_way, n_shot+15)[:, :n_shot]
                    glabels = glabels.contiguous().view(-1)
                    text_features = val_text[glabels]

                    _, sup_im_features,_,_,_ = model.forward_with_semantic_prompt(sup, text_features,args)
                    sup_im_features=sup_im_features.cuda()
                    sup_im_features = sup_im_features.view(n_train_way, n_shot, -1).mean(dim=1)
                    _,que_im_features,_,_,_= model(que,args)
                    
                    logits = F.normalize(que_im_features, dim=-1) @ F.normalize(sup_im_features, dim=-1).t()
                    acc = utils.compute_acc(logits, labels)

                    aves['va'].add(acc)

        # post #
        if lr_scheduler is not None:
            lr_scheduler.step(epoch-1)

        # key's value to item()
        for k, v in aves.items():
            aves[k] = v.item()

        # time of a epoch ,sum epochs and max epochs
        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)
        
        # log train loss and acc  
        log_str = 'epoch {}, train {:.4f}|{:.4f}'.format(epoch, aves['tl'], aves['ta'])
        writer.add_scalars('loss', {'train': aves['tl']}, epoch)
        writer.add_scalars('acc', {'train': aves['ta']}, epoch)
        
        # log val acc 
        if epoch % ef_epoch == 0 or epoch==1:
            log_str += ', val {}: {:.4f}'.format(n_shot,aves['va'])
            writer.add_scalars('acc', {'val': aves['va']}, epoch)
        log_str += ', {} {}/{}'.format(t_epoch, t_used, t_estimate)
        utils.log(log_str)

        # save checkpoint
        checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        if (save_epoch is not None) and epoch % save_epoch == 0:
            torch.save(checkpoint,os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))
        if aves['va'] > max_va:
            max_va = aves['va']
            torch.save(checkpoint, os.path.join(save_path, 'best.pth'))
        if epoch ==99:
            log_str='best: {}'.format(max_va)

        writer.flush()


if __name__ == '__main__':

    main(args)
