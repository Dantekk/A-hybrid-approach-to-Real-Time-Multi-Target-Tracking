from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import json
import torch
import torch.utils.data
from torchvision.transforms import transforms as T
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory


def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    print('Setting up data...')
    Dataset = get_dataset(opt.dataset, opt.task)
    f = open(opt.data_cfg)
    data_config = json.load(f)
    trainset_paths = data_config['train']
    dataset_root = data_config['root']
    ###
    valid_paths = data_config['test']
    ###
    f.close()
    transforms = T.Compose([T.ToTensor()])
    dataset = Dataset(opt, dataset_root, trainset_paths, (1088, 608), augment=True, transforms=transforms)
    ###
    valid_dataset = Dataset(opt, dataset_root, valid_paths, (1088, 608), augment=False, transforms=transforms)
    ###
    opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    print(opt)

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    print('Creating optimizer with lr : ',opt.lr)
    optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    start_epoch = 0

    # Get dataloader
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    ###
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )
    ###


    ### Dict for history
    print("******* ",opt.resume)
    if opt.resume and opt.pretrained==0:
        with open(opt.dir_base_save+'/history.json') as handle:
            history = json.loads(handle.read())
    else :
        history = {'loss_train': [], 'hm_loss_train': [], 'wh_loss_train': [], 'off_loss_train': [], 'id_loss_train': [],
               'loss_valid': [], 'hm_loss_valid': [], 'wh_loss_valid': [], 'off_loss_valid': [], 'id_loss_valid': []}


    print('Starting training...')
    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
    print("train_only_det : ",opt.train_only_det)
    print("pretrained : ", opt.pretrained)
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, trainer.optimizer, opt.resume, opt.lr, opt.lr_step, pretrained=0 if opt.pretrained==0 else 1,
            train_only_det=0 if opt.train_only_det==0 else 1)


    ####
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write('epoch train : {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))

        

        ### Aggiorna il train history
        history['loss_train'].append(log_dict_train['loss'])
        history['hm_loss_train'].append(log_dict_train['hm_loss'])
        history['wh_loss_train'].append(log_dict_train['wh_loss'])
        history['off_loss_train'].append(log_dict_train['off_loss'])
        history['id_loss_train'].append(log_dict_train['id_loss'])
        ###

        '''
        if opt.val_intervals > 0 : #and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer)
        '''
        #save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
        #           epoch, model, optimizer)
        #save_model('/home/ciro.panariello/tracking/FairMOT/FairMOT/src/results/save_model/res50/model_{}.pth'.format(mark),
        #           epoch, model, optimizer)

        logger.write('\n')
        if epoch in opt.lr_step:
            #save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
            #           epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        ### Valid phase
        log_dict_valid, _ = trainer.val(epoch, valid_loader)
        logger.write('epoch val : {} |'.format(epoch))
        for k, v in log_dict_valid.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        logger.write('\n')


        ### Decide ogni quanto salvare il modello
        if epoch % 1 == 0 or epoch >= 25:
            save_model(opt.dir_base_save+'/model_{}.pth'.format(epoch),
                     epoch, model, optimizer, save_classified_id=True if opt.train_only_det==0 else False)
            #save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
            #           epoch, model, optimizer)

        ### Aggiorna il valid history
        history['loss_valid'].append(log_dict_valid['loss'])
        history['hm_loss_valid'].append(log_dict_valid['hm_loss'])
        history['wh_loss_valid'].append(log_dict_valid['wh_loss'])
        history['off_loss_valid'].append(log_dict_valid['off_loss'])
        history['id_loss_valid'].append(log_dict_valid['id_loss'])
        ###


        ### Salva il dict
        with open(opt.dir_base_save+'/history.json', 'w') as fp:
            json.dump(history, fp)

    logger.close()


if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
