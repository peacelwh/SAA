import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--n_shot', type=int, default=5)
parser.add_argument('--phase', type=str, default='meta') 
parser.add_argument('--dataset', type=str, default='tieredImageNet', choices=['miniImageNet', 'tieredImageNet', 'CIFAR-FS','FG-CUB','FG-Dogs'])
parser.add_argument('--image_size', type=int,default=224)
parser.add_argument('--episode', type=int, default=1000)
parser.add_argument('--backbone', type=str, default='visformer-t')
parser.add_argument('--text_length', type=int,default=20)
parser.add_argument('--load_dir', type=str, default='meta_gla')
parser.add_argument('--load', default='')
parser.add_argument('--eqnorm', action='store_false')
parser.add_argument('--aug_support', type=int, default=20)
parser.add_argument('--stage', type=float, default=2.3, choices=[2, 2.1, 2.2, 2.3, 3, 3.1, 3.2, 3.3])
parser.add_argument('--num_workers', type=int, default='8', choices=[16,8,4,2,1])
parser.add_argument('--avg', type=str, default='all', choices=['all', 'patch', 'head'])
parser.add_argument('--gpu', default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import random
import numpy as np
from tqdm import tqdm
import scipy.stats
from sklearn.linear_model import LogisticRegression
from torch.utils.tensorboard import SummaryWriter

import clip

from datas.dataloader import EpisodeSampler
from datas.datasets_meta import Datasets
# from datas.js import Datasets
from model import visformer
import utils


def fix_random_seeds(seed=12345):
    """
    Fix random seeds.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h


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


def main(config):
    fix_random_seeds(args.seed)
    

    # fs_loder #
    args_dict = vars(args)
    train_dataset = Datasets(args.dataset, split='train', **args_dict)
    num_classes = len(train_dataset.dataset.classes)
    n_way = 5
    n_shot=args.n_shot
    fs_dataset=Datasets(args.dataset,split='test',**args_dict)
    episode_sampler = EpisodeSampler(fs_dataset.dataset.targets, 
                                    args.episode, 
                                    n_way, 
                                    n_shot + 15)    
    fs_loader = DataLoader(fs_dataset, batch_sampler=episode_sampler, num_workers=args.num_workers)
    if args.aug_support == 1:
        utils.log('fs dataset: {} (x{}), {}'.format(fs_dataset[0][0].shape, len(fs_dataset),len(fs_dataset.dataset.classes)))
    else:
        utils.log('fs dataset: {} (x{}), {}'.format(fs_dataset[0][0][0].shape, len(fs_dataset),len(fs_dataset.dataset.classes)))

    # model #
    # clip
    device=f"cuda:{0}"
    teacher, _ = clip.load("ViT-B/32",device=device)
    text_dim = 512  
    if args.text_length !=-1:
        teacher.context_length = args.text_length
        teacher.positional_embedding.data = teacher.positional_embedding.data[:args.text_length]
        for layer in teacher.transformer.resblocks:
            layer.attn_mask.data = layer.attn_mask.data[:args.text_length, :args.text_length]

    val_text = get_text_feature(teacher, fs_dataset,args)
    train_text=get_text_feature(teacher, train_dataset,args)

    # equal length normalization for features
    if args.eqnorm:
        avg_length = (train_text ** 2).sum(-1).sqrt().mean().item()
        val_text = F.normalize(val_text, dim=-1) * avg_length

    # backbone
    if args.backbone == 'visformer-t':
        model = visformer.visformer_tiny(num_classes=num_classes,phase=args.phase)
    else:
        raise ValueError(f'unknown model: {args.model}')
    
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
        
    # init ="directory/"+args.dataset+'/'+args.load
    init ="save/"+args.dataset+'/'+args.load_dir+'_{}shot'.format(args.n_shot)+'/'+'best.pth'
    checkpoint = torch.load(init,map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    model=model.cuda()
    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    # test #
    model.eval()
    aves_keys = []
    aves_keys += ['fsa-' + str(n_shot)]
    aves = {k: utils.Averager() for k in aves_keys}
    va_lst = []
    with torch.no_grad():

        for episode in tqdm(fs_loader,desc='fs-' + str(n_shot), leave=False):
            if args.aug_support ==1:
                image = episode[0].cuda()  # way * (shot+15)
                glabels = episode[1].cuda()
                labels = torch.arange(n_way).unsqueeze(-1).repeat(1, 15).view(-1).cuda()

                image = image.view(n_way, n_shot+15, *image.shape[1:])  #([5, 16, 3, 80, 80])
                sup, que = image[:, :n_shot].contiguous(), image[:, n_shot:].contiguous()
                sup, que = sup.view(-1, *sup.shape[2:]), que.view(-1, *que.shape[2:])

                glabels = glabels.view(n_way, n_shot + 15)[:, :n_shot]
                glabels = glabels.contiguous().view(-1)
                text_features = val_text[glabels]
                
                _, sup_im_features, logits_token,_, _ = model.forward_with_semantic_prompt(sup, text_features,args)
                sup_im_features=sup_im_features.cuda()
                sup_im_features = sup_im_features.view(n_way, n_shot, -1).mean(dim=1)
                _,que_im_features,_,_,_= model(que,args)
        
                logits = F.normalize(que_im_features, dim=-1) @ F.normalize(sup_im_features, dim=-1).t()
                acc = utils.compute_acc(logits, labels)

                aves['fsa-' + str(n_shot)].add(acc)
                va_lst.append(acc)

            else:
                image = torch.cat(episode[0]).cuda()  # aug_support * way * (shot+15)
                glabels = episode[1].cuda()
                labels = torch.arange(n_way).unsqueeze(-1).repeat(1, 15).view(-1).cuda()

                image = image.view(args.aug_support, n_way, n_shot + 15, *image.shape[1:])  #([2, 5, 16, 3, 80, 80])
                sup = image[:, :, :n_shot].contiguous().view(-1, *image.shape[3:])
                que = image[0, :, n_shot:].contiguous().view(-1, *image.shape[3:])

                glabels = glabels.view(n_way, n_shot + 15)[:, :n_shot]
                glabels = glabels.unsqueeze(0).repeat(args.aug_support, 1, 1).contiguous().view(-1)
                text_features = val_text[glabels]

                _, sup_im_features, logits_token,_, _ = model.forward_with_semantic_prompt(sup, text_features,args)
                sup_im_features=sup_im_features.cuda()
                _,que_im_features,_,_,_= model(que,args)

                # fc
                x_train = F.normalize(sup_im_features, dim=-1).cpu().numpy()
                y_train = torch.arange(n_way).unsqueeze(0).unsqueeze(-1).repeat(args.aug_support, 1, n_shot).view(-1).numpy()
                x_test = F.normalize(que_im_features, dim=-1).cpu().numpy()
                clf = LogisticRegression(penalty='l2',
                                            random_state=0,
                                            C=1.0,
                                            solver='lbfgs',
                                            max_iter=1000,
                                            multi_class='multinomial')
                clf.fit(x_train, y_train)
                pred = clf.predict(x_test)
                pred = torch.tensor(pred).cuda()

                acc= labels.eq(pred).sum().float().item() / labels.shape[0]
                aves['fsa-' + str(n_shot)].add(acc)
                va_lst.append(acc)
        
        
        print('test epoch : acc={:.2f} +- {:.2f} (%)'.format(
                aves['fsa-'+ str(n_shot)].item() * 100,
                mean_confidence_interval(va_lst) * 100))
        
        prediction_folder = os.path.join('./save', args.dataset, args.load_dir+'_{}shot'.format(args.n_shot))
    
        prediction_dir = os.path.join(prediction_folder, f"{args.load_dir}.txt")
        f_txt = open(prediction_dir, 'a')
        print('test acc-{}shot: {:.2f} +- {:.2f} (%)'.format(
            args.n_shot,
            aves['fsa-'+ str(n_shot)].item() * 100,
            mean_confidence_interval(va_lst) * 100),file=f_txt)   


if __name__ == '__main__':
    main(args) 
