import torch
import numpy as np

import argparse, json
import seaborn as sns
import matplotlib.pyplot as plt
import os, glob, sys, shutil
from time import time

from dataloader import DRIVE_folder
from unet.unet_model import UNet
import torch
torch.cuda.empty_cache()
from torchvision.utils import save_image
from PIL import Image

from dmt_trainer import getData_val, reconstruct_uncertainty_heatmap 

from unc_model import UncertaintyModel

def parse_func(args):
    ### Reading the parameters json file
    print("Reading params file {}...".format(args.params))
    with open(args.params, 'r') as f:
        params = json.load(f)

    activity = params['common']['activity']
    mydict = {}
    mydict['num_classes'] = int(params['common']['num_classes'])
    mydict['folder'] = [params['common']['img_folder'], params['common']['gt_folder']]
    mydict['segmodel_checkpoint_restore'] = params['common']['segmodel_checkpoint_restore']
    mydict['uncmodel_checkpoint_restore'] = params['common']['uncmodel_checkpoint_restore']
    mydict['dataname'] = params['common']['dataname']
    mydict['network'] = params['common']['network']
    
    mydict['gpu'] = False
    if params['common']['gpu'] == "true":
        mydict['gpu'] = True        
        
    if activity == "test":
        mydict['MCSamples'] = int(params['test']['MCSamples'])
        mydict['output_folder'] = params['test']['output_folder']

    else:
        print("Wrong activity chosen")
        sys.exit()

    return activity, mydict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, help="Path to the parameters file")
    parser.add_argument('--dataset', type= str, default = "DRIVE")   

    if len(sys.argv) == 1:
        print("Path to parameters file not provided. Exiting...")

    else:
        args = parser.parse_args()
        activity, mydict = parse_func(args)

    with open(args.params, 'r') as f:
        params = json.load(f)
    
    # call train
    print("Inference!")

    if mydict['gpu']:
        device = torch.device("cuda")
        print("CUDA device: {}".format(device))
        if not torch.cuda.is_available():
            print("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

    if not os.path.exists(mydict['output_folder']):
        os.makedirs(mydict['output_folder'])

    if not os.path.exists(os.path.join(mydict['output_folder'],'inputs')):
        shutil.copytree('inputs/', os.path.join(mydict['output_folder'],'inputs'))
    
    if not os.path.exists(os.path.join(mydict['output_folder'],'output')):
        os.makedirs(os.path.join(mydict['output_folder'], 'output'))

    # Test Data
    if mydict['dataname'].lower() == "drive":
        test_set = DRIVE_folder(mydict['folder'])
        n_channels = 3
        in_channels = 10
    
    elif mydict['dataname'].lower() == "fill-your-own-dataset":
        pass


    test_generator = torch.utils.data.DataLoader(test_set,batch_size=1,shuffle=False,num_workers=1, drop_last=False)

    if mydict['network'].lower() == "unet":
        feature_extractor = UNet(n_channels=n_channels, n_classes=mydict['num_classes'], start_filters=64)
    elif mydict['network'].lower() == "fill-your-own-network":
        pass

    binary_classifier = UncertaintyModel(in_channels=in_channels, num_features=36, hidden_units=48).float().to(device)

    if mydict['gpu']:
        feature_extractor = feature_extractor.to(device)
        binary_classifier = binary_classifier.to(device)
        
    if mydict['segmodel_checkpoint_restore'] != "":
        feature_extractor.load_state_dict(torch.load(mydict['segmodel_checkpoint_restore']), strict=True)
        print("loaded segmodel checkpoint! {}".format(mydict['segmodel_checkpoint_restore']))
    else:
        print("No seg model found!")
        sys.exit()

    if mydict['uncmodel_checkpoint_restore'] != "":
        binary_classifier.load_state_dict(torch.load(mydict['uncmodel_checkpoint_restore']), strict=True)
        print("loaded uncertainty model checkpoint! {}".format(mydict['uncmodel_checkpoint_restore']))
    else:
        print("No uncertainty model found!")
        sys.exit()

    print("Todo: {}".format(len(test_generator)))
    test_start_time = time()

    with torch.no_grad():

        binary_classifier.train() # for dropout
        feature_extractor.eval()

        test_iterator = iter(test_generator)
        for i in range(len(test_generator)):
            x, y_gt, filename = next(test_iterator)
            x = x.to(device, non_blocking=True)
            y_gt = y_gt.to(device, non_blocking=True)

            y_patchlikelihood = feature_extractor(x)

            imgbatch, unc_input, unc_gt = getData_val(mydict['num_classes'], mydict['output_folder'], x, y_patchlikelihood, y_gt)

            if unc_input is not None:

                if mydict['gpu']:
                    imgbatch = imgbatch.float().to(device)
                    unc_input = unc_input.float().to(device)
                else:
                    unc_input = unc_input.float()

                unc_pred_mu = []
                unc_pred_logvar = []
                for _ in range(mydict['MCSamples']):
                    temp = binary_classifier(imgbatch, unc_input) # contains both mu and log_var
                    unc_pred_mu.append(torch.squeeze(temp[0], dim=1).detach().cpu().numpy()) 
                    unc_pred_logvar.append(torch.squeeze(temp[1], dim=1).detach().cpu().numpy())

                print(torch.squeeze(y_gt).detach().cpu().numpy().shape)
                
                seg_map = reconstruct_uncertainty_heatmap(mydict['output_folder'], unc_pred_mu, unc_pred_logvar, torch.squeeze(y_gt).detach().cpu().numpy().shape, unc_gt.detach().cpu().numpy(),filename[0]) #seg_map is NCHW

            # seg_map is heatmap so save with cv2
            filename = filename[0]
            tempdir = os.path.join(mydict['output_folder'], filename.split('/')[-2])
            if not os.path.exists(tempdir):
                os.makedirs(tempdir)

            save_image(x, os.path.join(tempdir, 'img_' + filename.split('/')[-1]  + '.png'))
            save_image(torch.squeeze(y_gt*255), os.path.join(tempdir, 'gt_' + filename.split('/')[-1] + '.png'))

            ax = sns.heatmap(seg_map, cmap=plt.cm.coolwarm,vmin=0, vmax=1)
            ax.set_axis_off()

            plt.show()
            plt.savefig(os.path.join(tempdir, 'heatmap_x_' + filename.split('/')[-1] + '.png'), bbox_inches='tight', pad_inches=0)
            plt.clf()

            # save heatmap as npy and likelihood as binary --- to overlay later
            hm_npy = np.squeeze(np.clip(seg_map, 0., 1.))
            bb_npy =  y_patchlikelihood # backbone

            if mydict['num_classes'] == 1:
                bb_npy = torch.sigmoid(torch.squeeze(y_patchlikelihood)).detach().cpu().numpy()
                bb_npy = np.where(bb_npy >= 0.5, 1., 0.)

            elif mydict['num_classes'] == 2:
                bb_npy = torch.squeeze(torch.argmax(y_patchlikelihood, dim=1)).detach().cpu().numpy()

            bb_npy = (bb_npy*255.).astype(np.uint8)
            im_pred = Image.fromarray(bb_npy)

            im_pred.save(os.path.join(tempdir, 'backbone_' + filename.split('/')[-1] + '.png'))
            np.save(os.path.join(tempdir, 'heatmap_' + filename.split('/')[-1] + '.npy'), hm_npy)

            print("{} Done!".format(filename))
            