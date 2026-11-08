import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/home/customer5/d_1EUVP/config/config.yml',
                        help='yml file for configuration')
    parser.add_argument('-p', '--phase', type=str, help='Run train(training)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break

                # # 在1000000步后冻结第一阶段参数
                # if current_step == 1000000:
                #     logger.info('Freezing stage1 parameters at step 500000')
                #     for name, param in diffusion.netG.named_parameters():
                #         if 'enh_stage1' in name:   # 根据实际模型结构调整
                #             param.requires_grad = False

                # # 在前50000步只训练stage1参数
                # if current_step < 70000:
                #     # 冻结非stage1参数
                #     for name, param in diffusion.netG.named_parameters():
                #         if 'enh_stage1' not in name:  # 非stage1参数
                #             param.requires_grad = False
                #         else:  # stage1参数
                #             param.requires_grad = True
                
                # # 50000步后只训练非stage1参数
                # elif current_step == 70000:
                #     logger.info('Switching to train non-stage1 parameters at step 50000')
                #     for name, param in diffusion.netG.named_parameters():
                #         if 'enh_stage1' in name:  # stage1参数
                #             param.requires_grad = False
                #         else:  # 非stage1参数
                #             param.requires_grad = True

                if wandb_logger:
                    wandb_logger.log_metrics(len(train_data))
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_psnr = 0.0
                    avg_ssim = 0.0
                    avg_stage1_psnr = 0.0  
                    avg_stage1_ssim = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')
                    for _,  val_data in enumerate(val_loader):
                        idx += 1
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals()


                        restore_img = Metrics.tensor2img(visuals['output'])  # uint8
                        target_img = Metrics.tensor2img(visuals['target'])  # uint8
                        input_img = Metrics.tensor2img(visuals['input'])  # uint8
                        # a_img = Metrics.tensor2img(visuals['a'])  # uint8
                        # t_img = Metrics.tensor2img(visuals['t'])  # uint8

                        # generation
                        Metrics.save_img(target_img, '{}/{}_{}_taget.png'.format(result_path, current_step, idx))
                        Metrics.save_img(restore_img, '{}/{}_{}_output.png'.format(result_path, current_step, idx))
                        Metrics.save_img(input_img, '{}/{}_{}_input.png'.format(result_path, current_step, idx))
                        # Metrics.save_img(a_img, '{}/{}_{}a.png'.format(result_path, current_step, idx))
                        # Metrics.save_img(t_img, '{}/{}_{}t.png'.format(result_path, current_step, idx))
                        tb_logger.add_image(
                            'Iter_{}'.format(current_step),
                            np.transpose(np.concatenate((input_img, restore_img, target_img,), axis=1), [2, 0, 1]),idx)
                        avg_psnr += Metrics.calculate_psnr(restore_img, target_img)
                        avg_ssim += Metrics.calculate_ssim(restore_img, target_img) 
                        if wandb_logger:
                            wandb_logger.log_image(
                                f'validation_{idx}',
                                np.concatenate((input_img, restore_img, target_img), axis=1)
                            )

                    avg_psnr = avg_psnr / idx
                    avg_ssim = avg_ssim / idx
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    logger.info('# Validation # PSNR: {:.4e}, SSIM: {:.4f} '.format(
                        avg_psnr, avg_ssim))
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim: {:.4f}'.format(
                        current_epoch, current_step, avg_psnr, avg_ssim))
                    # tensorboard logger
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    tb_logger.add_scalar('ssim', avg_ssim, current_step)
                    # 新增stage1指标记录
                    tb_logger.add_scalar('stage1_psnr', avg_stage1_psnr, current_step)
                    tb_logger.add_scalar('stage1_ssim', avg_stage1_ssim, current_step)

                    if wandb_logger:
                        wandb_logger.log_metrics({'validation/val_psnr': avg_psnr,'validation/val_step': val_step})
                        val_step += 1

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        logger.info('End of training.')
    else:
        raise NotImplementedError('phase should be the train phase')

