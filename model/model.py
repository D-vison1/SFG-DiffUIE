import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
logger = logging.getLogger('base')


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))
        self.schedule_phase = None

        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        
        self.stage1_params = []
        self.unet_params = []
        for name, param in self.netG.named_parameters():
            if 'enh_stage1' in name:  
                self.stage1_params.append(param)
                logger.info(f'Stage1 parameter: {name}')  
            else:
                self.unet_params.append(param)

        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam([
                {'params': self.stage1_params, 'lr': opt['train']["optimizer"]["lr_stage1"]},
                {'params': self.unet_params, 'lr': opt['train']["optimizer"]["lr_unet"]}
])
           #同步学习率
            unet_optim = torch.optim.Adam([{'params': self.unet_params}])
            unet_optim.param_groups[0]['lr'] = opt['train']["optimizer"]["lr_unet"]  
            
            self.scheduler_unet = torch.optim.lr_scheduler.CosineAnnealingLR(
                unet_optim,
                T_max=opt['train']['n_iter'],
                eta_min=opt['train']["optimizer"].get("min_lr_unet", 1e-6)
            )
            self.log_dict = OrderedDict()
        self.load_network()
       # self.print_network()

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad()
        l_pix, l_pix1 = self.netG(self.data)
        total_loss = l_pix + 0.2 * l_pix1
        total_loss.backward()
        self.optG.step()

        # 更新UNet的学习率调度器
        self.scheduler_unet.step()
        self.optG.param_groups[1]['lr'] = self.scheduler_unet.get_last_lr()[0]
        # set log
        self.log_dict['l_pix'] = l_pix.item()
        self.log_dict['stage1_loss'] = l_pix1.item()
        self.log_dict['lr_unet'] = self.optG.param_groups[1]['lr']

    def test(self, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.Restoration,self.stage1 = self.netG.module.restore(
                    self.data['input'], continous)
            else:
                self.Restoration,self.stage1 = self.netG.restore(
                    self.data['input'], continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.Restoration.detach().float().cpu()
        else:

            out_dict['output'] = self.Restoration.detach().float().cpu()
            out_dict['input'] = self.data['input'].detach().float().cpu()
            out_dict['target'] = self.data['target'].detach().float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,'scheduler': self.scheduler_unet.state_dict(), 'optimizer': self.optG.state_dict(), 'stage1_frozen': iter_step >= 1000000}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path), strict=(not self.opt['model']['finetune_norm']))
            # network.load_state_dict(torch.load(
            #     gen_path), strict=False)
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
                 # recovery scheduler state
                if 'scheduler' in opt and opt['scheduler'] is not None:
                    self.scheduler_unet.load_state_dict(opt['scheduler'])
                    self.optG.param_groups[1]['lr'] = self.scheduler_unet.get_last_lr()[0]
                #  # 根据保存的状态恢复训练模式
                # if opt.get('stage1_only', False):  # 仅训练stage1阶段
                #     for name, param in self.netG.named_parameters():
                #         param.requires_grad = ('enh_stage1' in name)
                # elif opt.get('stage1_frozen', False):  # 冻结stage1阶段
                #     for name, param in self.netG.named_parameters():
                #         param.requires_grad = ('enh_stage1' not in name)
                # else:  # 默认情况
                #     for param in self.netG.parameters():
                #         param.requires_grad = True
             # 强制解除冻结（当iter_step < 500000）
            if self.begin_step < 1000000:
                for param in self.stage1_params:
                    param.requires_grad = True
            elif opt.get('stage1_frozen', False):  #不存在 'stage1_frozen' 这个键，则返回默认值 False
                for param in self.stage1_params:
                    param.requires_grad = False