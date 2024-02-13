import torch
import os, glob, sys
import numpy as np
from PIL import Image
from os.path import join as pjoin
from torchvision import transforms
from torch.utils import data

class DRIVE(data.Dataset):
    def __init__(self, listpath, folderpaths, task, crop_size = 128):

        self.listpath = listpath
        self.imgfolder = folderpaths[0]
        self.gtfolder = folderpaths[1]
        self.crop_size = crop_size

        self.dataCPU = {}
        self.dataCPU['image'] = []
        self.dataCPU['label'] = []
        self.dataCPU['filename'] = []

        self.task = task
 
        self.to_tensor = transforms.ToTensor() # converts HWC in [0,255] to CHW in [0,1]

        self.loadCPU()

    def loadCPU(self):
        with open(self.listpath, 'r') as f:
            mylist = f.readlines()
        mylist = [x.rstrip('\n') for x in mylist]

        for i, entry in enumerate(mylist):

            components = entry.split('.')
            filename = components[0]

            if self.task == "test":
                im_path = pjoin(self.imgfolder, filename) + '_test.tif'
            else:
                im_path = pjoin(self.imgfolder, filename) + '_training.tif'

            gt_path = pjoin(self.gtfolder, filename) + '_manual1.gif'
            img = Image.open(im_path)
            gt = Image.open(gt_path)

            img = self.to_tensor(img) 
            gt = self.to_tensor(gt)

            # normalize within a channel
            for j in range(img.shape[0]):
                meanval = img[j].mean()
                stdval = img[j].std()
                img[j] = (img[j] - meanval) / stdval

            # cpu store
            self.dataCPU['image'].append(img)
            self.dataCPU['label'].append(gt)
            self.dataCPU['filename'].append(filename)

    def __len__(self): # total number of 2D slices
        return len(self.dataCPU['filename'])

    def __getitem__(self, index): # select random crop and return CHW torch tensor

        torch_img = self.dataCPU['image'][index] #HW
        torch_gt = self.dataCPU['label'][index] #HW

        if self.task == "train":
            # crop: compute top-left corner first
            _, H, W = torch_img.shape
            corner_h = np.random.randint(low=0, high=H-self.crop_size)
            corner_w = np.random.randint(low=0, high=W-self.crop_size)

            torch_img = torch_img[:, corner_h:corner_h+self.crop_size, corner_w:corner_w+self.crop_size]
            torch_gt = torch_gt[:, corner_h:corner_h+self.crop_size, corner_w:corner_w+self.crop_size]

        return torch_img, torch_gt, self.dataCPU['filename'][index]



# uses folder as input instead of csv
class DRIVE_folder(data.Dataset):
    def __init__(self, folderpaths):

        self.imgfolder = folderpaths[0]
        self.gtfolder = folderpaths[1]
        self.suffix = ".tif"

        self.dataCPU = {}
        self.dataCPU['image'] = []
        self.dataCPU['label'] = []
        self.dataCPU['filename'] = []

        self.indices = [] 
        self.to_tensor = transforms.ToTensor()

        self.loadCPU()

    def loadCPU(self):
        mylist = glob.glob(self.imgfolder + "/*" + self.suffix)
        subdir = False
        if len(mylist) == 0:
            subdir = True
            mylist = glob.glob(self.imgfolder + "/*/*" + self.suffix)

        assert len(mylist) != 0

        mylist.sort()        

        for i, im_path in enumerate(mylist):
            #gt_path = pjoin(self.gtfolder, filename) + '_manual1.gif'
            fname = im_path.replace(self.suffix, ".png").split('/')[-1]
            fname = "gt_" + im_path.replace("_test", "").split('/')[-2] + '/' + fname
            gt_path = glob.glob(self.gtfolder + "/" + fname)

            assert len(gt_path) == 1
            gt_path = gt_path[0]

            img = Image.open(im_path)
            gt = np.array(Image.open(gt_path))[:,:,0]/255.

            img = self.to_tensor(img)
            gt = torch.from_numpy(gt)

            #normalize within a channel
            for j in range(img.shape[0]):
                meanval = img[j].mean()
                stdval = img[j].std()
                img[j] = (img[j] - meanval) / stdval

            self.indices.append((i))

            #cpu store
            self.dataCPU['image'].append(img)
            self.dataCPU['label'].append(gt)
            self.dataCPU['filename'].append(im_path.split('/')[-2] + '/' + im_path.split('/')[-1].replace(self.suffix,""))

    def __len__(self): # total number of 2D slices
        return len(self.indices)

    def __getitem__(self, index): # return CHW torch tensor
        index = self.indices[index]
        #print("Doing {}".format(self.dataCPU['filename'][index]))
        return self.dataCPU['image'][index], self.dataCPU['label'][index], self.dataCPU['filename'][index]




if __name__ == "__main__":
    flag = "training"

    dst = DRIVE('/data/saumgupta/simple-unet-2d/datalists/val-list.csv', ["/data/saumgupta/simple-unet-2d/data/DRIVE/training/images","/data/saumgupta/simple-unet-2d/data/DRIVE/training/1st_manual"], task="val", crop_size=128) 
    training_generator = data.DataLoader(dst, shuffle=False, batch_size=2, num_workers=8)


    for step, (patch, mask, _) in enumerate(training_generator):
        pass
    print("One epoch done; steps: {}".format(step))