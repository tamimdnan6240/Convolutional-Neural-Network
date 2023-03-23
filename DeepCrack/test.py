from data.dataset import readIndex, dataReadPip, loadedDataset
from model.deepcrack import DeepCrack
from trainer import DeepCrackTrainer
import cv2
from tqdm import tqdm
import numpy as np
import torch
import os
import segmentation_models_pytorch as smp ## Tamim: added this package for model evaluation

# os.environ["CUDA_VISIBLE_DEVICES"] = '0' #Tamim: off visdom


def test(test_data_path="data/test_example_proposed.txt",
         save_path='deepcrack_results/',
         pretrained_model='./checkpoints_1/DeepCrack_CT260_FT1/checkpoints/DeepCrack_CT260_FT1_model_0000001_2023-03-20-03-27-39.pth', ):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    test_pipline = dataReadPip(transforms=None)

    test_list = readIndex(test_data_path)

    test_dataset = loadedDataset(test_list, preprocess=test_pipline)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                              shuffle=False, num_workers=1, drop_last=False)
    
    def dice_coefficient(pred, mask):
        smooth = 1e-5 #this is constant value to avoid divided by 0 error
        ## Flatten tensor
        pred = pred.view(-1)
        mask = mask.view(-1)
     # calculate intersection and union
        intersection = (pred * mask).sum()
        union = pred.sum() + mask.sum()
     # calculate dice cofficient 
        accuracy = (2 * intersection + smooth) / (union + smooth) ## 2 is coefficient, included in numerator to ensure dice coefficent is between o and 1
     ## accuracy means dice here  
        return accuracy

    # -------------------- build trainer --------------------- #

    device = torch.device("cuda")
    num_gpu = torch.cuda.device_count()

    model = DeepCrack()

    model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
    model.to(device)

    trainer = DeepCrackTrainer(model).to(device)

    model.load_state_dict(trainer.saver.load(pretrained_model, multi_gpu=True))

    model.eval()
    
    with torch.no_grad():
        for names, (img, lab) in tqdm(zip(test_list, test_loader)):
         test_data, test_target = img.type(torch.FloatTensor).to(device), lab.type(torch.FloatTensor).to(device)
         test_pred = trainer.val_op(test_data, test_target)
         test_pred = torch.sigmoid(test_pred[0].cpu().squeeze())
         # normalize predicted values to [0,1]
         test_pred = (test_pred - test_pred.min()) / (test_pred.max() - test_pred.min())
         # convert to uint8 and scale to [0,255]
         test_pred = (test_pred * 255)

         ## normalize and scale lab tensors
         lab_norm = (lab - lab.min()) / (lab.max() - lab.min())
         lab_scaled = (lab_norm * 255)

         ## print accuracy 
         dice = dice_coefficient(test_pred, lab_scaled)
         print("testing accuracy:", dice)

         ## print image and lab on same image
         save_pred = torch.zeros(512*2, (512))
         save_pred[:512,:] = test_pred
         save_pred[512:,:] = lab_scaled.cpu().squeeze()
         save_pred = save_pred.numpy() 

         ## concatenate test_pred and lab_tensors vertically

         # save output image
         save_name = os.path.join(save_path, os.path.split(names[1])[1])
         cv2.imwrite(save_name, save_pred.astype(np.uint8))
         


    ## Tamim: Model evaluation on test dataset
    ## Tamim: reference: https://www.kaggle.com/code/balraj98/unet-for-building-segmentation-pytorch 
    ## Tamim: if the indent is not alinged with this line the training function will not detect the lines; 


if __name__ == '__main__':
    test()
