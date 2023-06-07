from data.augmentation import augCompose,rotate,centerCrop, RandomFlip, RandomBlur, RandomColorJitter
from data.dataset import readIndex, dataReadPip, loadedDataset
from tqdm import tqdm
from model.deepcrack import DeepCrack
from trainer import DeepCrackTrainer
from config import Config as cfg
import numpy as np
import torch
import torchvision
from torchvision import transforms
import os
import cv2
import sys
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torch.utils.data import random_split ## Tamim: to splite train and validation

### Tamim closed it: os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id


def main():
    # ----------------------- dataset ----------------------- #

    data_augment_op = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.ColorJitter(brightness=(0.5,1.5),contrast=(1),saturation=(0.5,1.5),hue=(-0.1,0.1)),
    transforms.GaussianBlur(kernel_size=(7, 13), sigma=(0.1, 0.2)),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])
    train_pipline = dataReadPip(transforms=data_augment_op)

    test_pipline = dataReadPip(transforms=None)

    train_dataset = loadedDataset(readIndex(cfg.train_data_path, shuffle=True), preprocess=train_pipline)

    ## train_validation_split

    train = int(0.9 * len(train_dataset))
    
    val = int(len(train_dataset) - train)

    train_dataset, val_dataset = random_split(train_dataset, [train, val]) 

    test_dataset = loadedDataset(readIndex(cfg.val_data_path), preprocess=test_pipline) ## Tamim: Test_data_path is missing in original

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train_batch_size,
                                               shuffle=True, num_workers=4, drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.val_batch_size,
                                             shuffle=True, num_workers=4, drop_last=True)

 

    # -------------------- build trainer --------------------- #

    device = torch.device("cuda")
    num_gpu = torch.cuda.device_count()

    model = DeepCrack()
    model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
    model.to(device)

    trainer = DeepCrackTrainer(model).to(device)
    
    ## Tamim: ## Tamim: create 4 lists and append epoch_numbers, average_train_loss, average_val_loss, average_train_acc, avegage_val_acc 
    total_train_loss = []
    total_val_loss = []
    total_train_acc = []
    total_val_acc = []
    epoch_numbers = []

    if cfg.pretrained_model:
        pretrained_dict = trainer.saver.load(cfg.pretrained_model, multi_gpu=True)
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        ## Tamim removed it: trainer.vis.log('load checkpoint: %s' % cfg.pretrained_model, 'train info')

    try:

        for epoch in range(1, cfg.epoch):
            model.train()             ## Tamim Changed: trainer.vis.log('Start Epoch %d ...' % epoch, 'train info')
            # ---------------------  training ------------------- #
            bar = tqdm(enumerate(train_loader), total=len(train_loader))
            bar.set_description('Epoch %d --- Training --- :' % epoch)
            for idx, (img, lab) in bar:
                data, target = img.type(torch.cuda.FloatTensor).to(device), lab.type(torch.cuda.FloatTensor).to(device)
                pred = trainer.train_op(data, target)
                
                ## Tamim: pass data, pred to sigmoid function 
                data = data.cpu() 
                target = target.unsqueeze(1).cpu()
                pred[0] = torch.sigmoid(pred[0].contiguous().cpu())
                pred[1] = torch.sigmoid(pred[1].contiguous().cpu())
                pred[2] = torch.sigmoid(pred[2].contiguous().cpu())
                pred[3] = torch.sigmoid(pred[3].contiguous().cpu())
                pred[4] = torch.sigmoid(pred[4].contiguous().cpu())
                pred[5] = torch.sigmoid(pred[5].contiguous().cpu()) 

                # Tamim: display the augmented images in loop (line 112 to line 116)

                #-----------tamim: visualize batch size after augmenation-----------------------
                # get a batch of data
                # images, masks = next(iter(train_loader))

                # make the grid of augmented images
                
               # augmented_images = vutils.make_grid(images, nrow=4, normalize=True, scale_each=True)
                #make the grid of augmented masks
 
                # augmented_masks = vutils.make_grid(masks, nrow=4, normalize=True, scale_each=True)

                # plt.figure(figsize=(8, 8))
                # plt.imshow(augmented_images.permute(1, 2, 0))
                # plt.axis("off")
                # plt.show()
                # plt.savefig(f"Epoch_{epoch}.png")
                
            ## Tamim_comment: cfg.vis_train_loss_every was as condition to visualize. We don't need that. Like other CNN training, we can 
            # pass our model data and target to loss function, then we calculate training loss and training accuracy, so sotre 
            # all losses to 'trainer.log_loss'
            ## Tamim removed it: if idx % cfg.vis_train_loss_every == 0:trainer.vis.log(trainer.log_loss, 'train_loss'). trainer.vis.plot_many({
            ## Tamim_comment: in visdom.py img 
            
            trainer.log_loss['train_total_loss']: trainer.log_loss['total_loss']
            trainer.log_loss['train_output_loss']: trainer.log_loss['output_loss']
            trainer.log_loss['train_fuse5_loss']: trainer.log_loss['fuse5_loss']
            trainer.log_loss['train_fuse4_loss']: trainer.log_loss['fuse4_loss']
            trainer.log_loss['train_fuse3_loss']: trainer.log_loss['fuse3_loss']
            trainer.log_loss['train_fuse2_loss']: trainer.log_loss['fuse2_loss']
            trainer.log_loss['train_fuse1_loss']: trainer.log_loss['fuse1_loss']
            
            ## Tamim: printed trainer.log_loss
            print(trainer.log_loss, 'train_log_loss')
            
            # Tamim:-----------------------average losses from total train loss----------------                
            ## Tamim: training_loss
            training_loss = trainer.log_loss['total_loss']
            print(f'Epoch {epoch} \t\t total_training_loss: {training_loss}')  ## Tamim: this for both every and many, but cosider for total
            total_train_loss.append(training_loss)

            ## Tamim removed it, see line 67-69: if idx % cfg.vis_train_acc_every == 0:
            
            #  Tamim removed it, see line 67-69, trainer.acc_op(pred[0], target)
            #  Tamim removed it, see line 67-69, trainer.vis.log(trainer.log_acc, 'train_acc')
            #  Tamim removed it, see line 67-69, trainer.vis.plot_many({
            #  Tamim removed it, see line 67-69, 
            
            trainer.acc_op(pred[0], target)
            trainer.log_acc['train_mask_acc']: trainer.log_acc['mask_acc'] 
            trainer.log_acc['train_mask_pos_acc']: trainer.log_acc['mask_pos_acc']
            trainer.log_acc['train_mask_neg_acc']: trainer.log_acc['mask_neg_acc'] 

 
            
            
            
            ## Tamim: stored sigmoids in train_acc but just print mask_acc to see accuracy

            print(trainer.log_acc, 'train_log_acc')
            training_acc = trainer.log_acc['mask_acc']

            print(f'Epoch {epoch} \t\t total_training_accuracy: {training_acc}')
            total_train_acc.append(training_acc)
            
            ## Tamim: printed trainer.log_Acc

            #  Tamim removed it, see line 67-69,if idx % cfg.val_every == 0:
            #  Tamim removed it, see line 67-69,        trainer.vis.log('Start Val %d ....' % idx, 'train info')

                    # -------------------- val ------------------- #

            model.eval()
            val_loss = {
                        'eval_total_loss': 0,
                        'eval_output_loss': 0,
                        'eval_fuse5_loss': 0,
                        'eval_fuse4_loss': 0,
                        'eval_fuse3_loss': 0,
                        'eval_fuse2_loss': 0,
                        'eval_fuse1_loss': 0,
                    }
            val_acc = {
                        'mask_acc': 0,
                        'mask_pos_acc': 0,
                        'mask_neg_acc': 0,
                    }

            bar.set_description('Epoch %d --- Evaluation --- :' % epoch)

           ## Tamim_comment: remember that torch.no_grad is for validation to clear the gradients of previous iteration
           
            with torch.no_grad():
                        for idx, (img, lab) in enumerate(val_loader, start=1):
                            val_data, val_target = img.type(torch.cuda.FloatTensor).to(device), lab.type(
                                torch.cuda.FloatTensor).to(device)
                            val_pred = trainer.val_op(val_data, val_target)
                            trainer.acc_op(val_pred[0], val_target)
## Tamim removed visdom and passing validation data, validation pred, and validation target to sigmoid function and store to val_acc
 ## Tamim: Here pred where data and target are taken, should be pass through sigmoid function to see the nonlinearity charateriestics                            
                            val_data = val_data.cpu()
                            val_target =  val_target.unsqueeze(1).cpu() 
                            val_pred[0] = torch.sigmoid(val_pred[0].contiguous().cpu())
                            val_pred[1] = torch.sigmoid(val_pred[1].contiguous().cpu())
                            val_pred[2] = torch.sigmoid(val_pred[2].contiguous().cpu())
                            val_pred[3] = torch.sigmoid(val_pred[3].contiguous().cpu())
                            val_pred[4] = torch.sigmoid(val_pred[4].contiguous().cpu())
                            val_pred[5] = torch.sigmoid(val_pred[5].contiguous().cpu())

 ## Here idx is the number of batch, it start with 0, if batch is 16, then at the end of each loop value will be 16 (equal to batch size)                           
                            ## Tamim: validation loss is defined
                            val_loss['eval_total_loss'] += trainer.log_loss['total_loss']
                            val_loss['eval_output_loss'] += trainer.log_loss['output_loss']
                            val_loss['eval_fuse5_loss'] += trainer.log_loss['fuse5_loss']
                            val_loss['eval_fuse4_loss'] += trainer.log_loss['fuse4_loss']
                            val_loss['eval_fuse3_loss'] += trainer.log_loss['fuse3_loss']
                            val_loss['eval_fuse2_loss'] += trainer.log_loss['fuse2_loss']
                            val_loss['eval_fuse1_loss'] += trainer.log_loss['fuse1_loss']
                                                     
                            ## Tamim_comment: validation accuracy is defined 
                            val_acc['mask_acc'] += trainer.log_acc['mask_acc']
                            val_acc['mask_pos_acc'] += trainer.log_acc['mask_pos_acc']
                            val_acc['mask_neg_acc'] += trainer.log_acc['mask_neg_acc']
                                                        
                           ## Tamim removed: trainer.vis.plot_many({, see line 146

## Tamim comment: (Line 162 - 167), here validation loss is divided by the iterator. it is the default code. 
## reason: as eval_total_loss and accuracy_mask 0 in validation, so new value with summation need to be divided by the batch
# size to find the varage validation and average accuracy. in orginal code, : is used instead of =,
# here : is not working, so we use =. According to basic coding it is right. 
                           
                            val_loss['eval_total_loss'] = val_loss['eval_total_loss'] / idx
                            val_loss['eval_output_loss'] = val_loss['eval_output_loss'] / idx
                            val_loss['eval_fuse5_loss'] = val_loss['eval_fuse5_loss'] / idx
                            val_loss['eval_fuse4_loss'] = val_loss['eval_fuse4_loss'] / idx
                            val_loss['eval_fuse3_loss'] = val_loss['eval_fuse3_loss'] / idx
                            val_loss['eval_fuse2_loss'] = val_loss['eval_fuse2_loss'] / idx
                            val_loss['eval_fuse1_loss'] = val_loss['eval_fuse1_loss'] / idx
                            
                            ## Tamim: print and store val_log_loss
                            
                            print(val_loss, "val_log_loss")  
                            
                            total_validation_loss = val_loss['eval_total_loss']
                            
                            print(f'Epoch {epoch} \t\t total_validation_loss: {total_validation_loss}')
                            
                            total_val_loss.append(total_validation_loss) ## Tamim: appended val_loss
                            
                            ## Tamim: print and store val_log_loss

                            ## Tamim removed, trainer.vis.plot_many({,  see line 146
                            
                             
                            val_acc['mask_acc'] = val_acc['mask_acc'] / idx
                            val_acc['mask_neg_acc'] = val_acc['mask_neg_acc'] / idx
                            val_acc['mask_pos_acc'] = val_acc['mask_pos_acc'] / idx
                           
                           ## Tamim: print and store val_log_acc
                            print(val_acc, "val_log_acc") ## Tamim: validation accuracy per batch (idx iterates per batch) 
                            
                            total_validation_acc = val_acc['mask_acc']
                            
                            print(f'Epoch {epoch} \t\t total_validation_accuracy: {total_validation_acc}')
                            
                            total_val_acc.append(total_validation_acc) ## Tamim: appended val_acc
                            
                            ## Tamim: appended epoch numbers   
                            epoch_numbers.append(epoch)

                        ## ----     plots loss and accuracies) .................        
                        #   Plot the average training and validation loss curves
                            print(epoch_numbers, 'epoch_numbers')
                            print(total_train_loss, 'total_train_loss')
                            print(total_val_loss, 'total_val_loss')
                            print(total_train_acc, "total_train_acc")
                            print(total_val_acc, 'total_val_acc')

                            
                            # ----------------- save model ---------------- #
                            if cfg.save_pos_acc < (val_acc['mask_pos_acc'] / idx) and cfg.save_acc < (
                                    val_acc['mask_acc'] / idx):
                                cfg.save_pos_acc = (val_acc['mask_pos_acc'] / idx)
                                cfg.save_acc = (val_acc['mask_acc'] / idx)
                                trainer.saver.save(model, tag='epoch(%d)' % (epoch))
                                # trainer.saver.save(model, tag='model') Tamim: Don't overwirite
                                # Tamim removed everything, just model will be saved if the saving condition meet and overwite the previous one
                                # trainer.vis.log('Save Model %s_epoch(%d)_acc(%0.5f/%0.5f)'
                                # % (cfg.name, epoch, cfg.save_pos_acc, cfg.save_acc), 'train info')

                            bar.set_description('Epoch %d --- Training --- :' % epoch)
                            model.train()

            if epoch != 0:
                # Tamim: removed visdom: trainer.saver.save(model, tag='model')
                # Tamim: removed visdom:trainer.vis.log('Save Model -%s_epoch(%d)' % (
                print(cfg.name, epoch, 'train info') ## Tamim: if the condition meet, then print train_info

    except KeyboardInterrupt:

        trainer.saver.save(model, tag='Auto_Save_Model')
        print('\n Catch KeyboardInterrupt, Auto Save final model : %s' % trainer.saver.show_save_pth_name)
        ## Tamim removed visdom, trainer.vis.log('Catch KeyboardInterrupt, Auto Save final model : %s' % trainer.saver.show_save_pth_name,
        ## Tamim removed visdom,               'train info')
        ## Tamim removed visdom,trainer.vis.log('Training End!!')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


if __name__ == '__main__':
    main()
