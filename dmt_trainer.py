import subprocess

import sys
import numpy as np
import os, shutil
import time
import torch
import csv
from PIL import Image
from torchvision import transforms

t0 = time.time()

DIPHA_CONST = 8067171840
DIPHA_IMAGE_TYPE_CONST = 1
DIM = 3

# img options for conv2d layer
mapHW = 64 # keeping this as half of patchsize
halfHw = int(mapHW/2)

s_gaussiid = 0.01 # sigma for gaussiid

a_rw = 0.2 # alpha for random walk

neighbors_8 = [[1,-1],[1,0],[1,1],[0,-1],[0,1],[-1,-1],[-1,0],[-1,1]]
neighbors_4 = [[1,0],[0,-1],[0,1],[-1,0]]
neighbors = neighbors_4

def dmt_2d(patch, Th):
    dipha_diagram_filename = os.path.join(savedir, 'inputs/diagram.bin')
    dipha_output_filename = os.path.join(savedir, 'inputs/complex.bin')
    vert_filename = os.path.join(savedir, 'inputs/vert.txt')
    dipha_edge_filename = os.path.join(savedir, 'inputs/dipha.edges')
    dipha_edge_txt = os.path.join(savedir, 'inputs/dipha_edges.txt')
    dipha_output = os.path.join(savedir, 'output/')

    nx, ny = patch.shape
    nz = 1
    im_cube = np.zeros([nx, ny, nz])
    im_cube[:, :, 0] = patch

    with open(dipha_output_filename, 'wb') as output_file:
        np.int64(DIPHA_CONST).tofile(output_file)
        np.int64(DIPHA_IMAGE_TYPE_CONST).tofile(output_file)
        np.int64(nx * ny * nz).tofile(output_file)
        np.int64(DIM).tofile(output_file)
        np.int64(nx).tofile(output_file)
        np.int64(ny).tofile(output_file)
        np.int64(nz).tofile(output_file)
        for k in range(nz):
            sys.stdout.flush()
            for j in range(ny):
                for i in range(nx):
                    val = int(-im_cube[i, j, k]*255)
                    np.float64(val).tofile(output_file)
        output_file.close()

    with open(vert_filename, 'w') as vert_file:
        for k in range(nz):
            sys.stdout.flush()
            for j in range(ny):
                for i in range(nx):
                    vert_file.write(str(i) + ' ' + str(j) + ' ' + str(k) + ' ' + str(int(-im_cube[i, j, k] * 255)) + '\n')
        vert_file.close()

    subprocess.call(["mpiexec", "-n", "1", "dipha-graph-recon/build/dipha", str(dipha_output_filename), str(dipha_diagram_filename), str(dipha_edge_filename), str(nx), str(ny), str(nz)])


    subprocess.call(["src/loop.out", str(dipha_edge_filename), str(dipha_edge_txt)]) 
    #pdb.set_trace()
    subprocess.call(["src/manifold.out", str(vert_filename), str(dipha_edge_txt), str(Th), str(dipha_output)])


# patch is a likelihood map in [0,1] range and shape NCHW
# Th is threshold value in [0,1] range
def dmt(num_classes, patch, Th=0.02):
    B, C, H, W = patch.shape
    if num_classes == 2:
        patch = np.array(patch.detach().cpu())[:,1,:,:] # probabilities in channel 1
    else:
        patch = np.array(patch.detach().cpu())[:,0,:,:]
    patch = np.expand_dims(patch,axis=1)
    
    for i in range(B):
        dmt_2d(patch[i,0,:,:], Th * 255)


def interpolate(nparr):
    omin = 0.0
    omax = 1.0
    imin  = np.min(nparr)
    imax = np.max(nparr)

    if imax == imin:
        return nparr

    return (nparr-imin)*(omax-omin)/(imax-imin) + omin

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def clip(x):
    return np.clip(x, 0., 1.)

# even though sigma is fixed, since the noise is IID, the resulting map is different every time
def gaussianIID(inp):
    return inp + np.random.normal(loc=0., scale=s_gaussiid, size=inp.shape)

def getdist(srcc,dstc):
    ans = np.sqrt(pow(srcc[0]-dstc[0],2) + pow(srcc[1]-dstc[1],2))
    return 1./ans

def checkbounds(curc, mapshape):
    if curc[0] >= mapshape[0] or curc[1] >= mapshape[1]:
        return False
    return True

# if walks backwards, stuck in loop, keeping a limit of 50 steps
def getPath(likelihood_map, srcc, dstc): 
    mini_image = np.zeros_like(likelihood_map)
    lmshape = likelihood_map.shape
    curc = srcc # current-coord
    mini_image[curc[0],curc[1]] = 1

    path_cnt = 0

    while(np.any(curc != dstc)):
        max_p_val = None
        neighbor_coord = None 

        for idx, offset in enumerate(neighbors):
            newc = curc + np.array(offset)

            if np.all(newc == dstc):
                neighbor_coord = newc
                break 
                
            if checkbounds(newc, lmshape):
                p_val = a_rw*getdist(newc,dstc) + (1.-a_rw)*likelihood_map[newc[0],newc[1]]

                if max_p_val is None or max_p_val < p_val:
                    max_p_val = p_val
                    neighbor_coord = newc

        curc = neighbor_coord
        mini_image[curc[0], curc[1]] = 1

        path_cnt+=1
        if path_cnt > 50:
            break

    return mini_image    


# img (CHW, C=3 rgb) and likelihood (HW) and dmt_binary_img (HW) are numpy
def getImgBatch(img, likelihood, dmt_bimg, srccoord, dstcoord): # return CHW 
    global del_cnt

    if np.random.rand() > 0.5: # choosing original likelihood map with 50% probability
        temp_lm = likelihood
        temp_path = dmt_bimg
    else:
        temp_lm = gaussianIID(likelihood)
        temp_path = getPath(temp_lm,srccoord,dstcoord)


    nstack = np.stack([temp_lm, temp_path])
    nstack = np.concatenate((img,nstack), axis=0)

    # crop to mapHW
    minx = min(srccoord[0],dstcoord[0])
    maxx = max(srccoord[0],dstcoord[0])
    miny = min(srccoord[1],dstcoord[1])
    maxy = max(srccoord[1],dstcoord[1])
    midx = int ( minx + (maxx - minx)/2 )
    midy = int ( miny + (maxy - miny)/2 )

    dstx = midx + halfHw
    dsty = midy + halfHw
    srcx = midx - halfHw
    srcy = midy - halfHw

    if dstx >= likelihood.shape[0]:
        dstx = likelihood.shape[0]
        srcx = dstx - mapHW

    if dsty >= likelihood.shape[1]:
        dsty = likelihood.shape[1]
        srcy = dsty - mapHW

    if srcx < 0:
        srcx = 0
        dstx = srcx + mapHW
    
    if srcy < 0:
        srcy = 0
        dsty = srcy + mapHW    

    return torch.from_numpy(nstack[:, srcx:dstx, srcy:dsty]) # CHW


# need to return img_batch as well which is NCHW (N=num structures)
def getManifoldFeatures(num_classes, img, likelihood, gt):
    return_input = []
    return_imgbatch = [] #use .stack on it later
    if gt is None: 
        return_gt = None
    else:
        return_gt = []

    manifold_filepath = os.path.join(savedir, "output/dimo_manifold.txt")
    vert_filepath = os.path.join(savedir, "output/dimo_vert.txt")

    if num_classes == 2:
        likelihood = torch.squeeze(likelihood).detach().cpu().numpy()[1] # probabilities in channel 1
    else:
        likelihood = torch.squeeze(likelihood).detach().cpu().numpy()
    img = torch.squeeze(img,0).detach().cpu().numpy() # CHW (C=3 for DRIVE; C=1 for ROSE)

    if gt is not None:
        gt = torch.squeeze(gt).detach().cpu().numpy()

    nx, ny = likelihood.shape

    vert_info = np.loadtxt(vert_filepath)
    bin_image = np.zeros([nx, ny])
    pers_image = np.zeros([nx, ny])
    likeli_image = np.zeros([nx, ny])
    srccoord = None 
    dstcoord = None

    manifold_cnt = -1
    with open(manifold_filepath, 'r') as manifold_info:
        reader = csv.reader(manifold_info, delimiter=' ')
        for row in reader:
            if len(row) != 3:
                if bin_image.sum() != 0:
                    manifold_cnt += 1
                    
                    # add to return_input and return_gt here
                    likeli_image = bin_image * likelihood
                    manifold_size = bin_image.sum()

                    dstcoord = [int(vert_info[v1,0]), int(vert_info[v1,1])]

                    return_imgbatch.append(getImgBatch(img, likelihood, bin_image, np.array(srccoord), np.array(dstcoord))) # returns torch tensor

                    return_input.append(np.array([manifold_size, likeli_image.sum()/manifold_size,pers_image.sum()/manifold_size,0.02]))

                    if gt is not None:
                        gt_manifold_size = (bin_image * gt).sum()

                        gt_label = gt_manifold_size/manifold_size

                        return_gt.append(gt_label)

                bin_image = np.zeros([nx, ny])
                pers_image = np.zeros([nx, ny])
                srccoord = None
                dstcoord = None
                continue

            v0 = int(row[0])
            v1 = int(row[1])
            pers_value = int(row[2])/255. # in [0,255] range

            if srccoord is None:
                srccoord = [int(vert_info[v0,0]), int(vert_info[v0,1])]

            bin_image[int(vert_info[v0,0]), int(vert_info[v0,1])] = 1
            bin_image[int(vert_info[v1,0]), int(vert_info[v1,1])] = 1

            pers_image[int(vert_info[v0,0]), int(vert_info[v0,1])] = pers_value
            pers_image[int(vert_info[v1,0]), int(vert_info[v1,1])] = pers_value

    if gt is not None:
        return_gt = torch.from_numpy(np.array(return_gt))

    if return_input == []:
        return_input = None
        return_imgbatch = None
    else:
        return_input = torch.from_numpy(np.array(return_input))
        return_imgbatch = torch.stack(return_imgbatch, dim=0) #NCHW form

    return return_imgbatch, return_input, return_gt # torch datatype


# likelihood is NCHW torch and cuda
def getData(num_classes, img, likelihood, gt):
    starttime = time.time()
    likelihood = torch.clamp(torch.sigmoid(likelihood),0.,1.)

    dmt(num_classes,likelihood)

    img_batch, unc_input, unc_gt = getManifoldFeatures(num_classes, img, likelihood, gt)
    #print("getData took {} minutes".format((time.time()-starttime)/60.))
    
    return img_batch, unc_input, unc_gt


def getData_train(num_classes, savedir_local, img, likelihood, gt):
    global savedir
    savedir = savedir_local
    return getData(num_classes, img, likelihood, gt)


def getData_val(num_classes, savedir_local, img, likelihood, gt):
    global savedir
    savedir = savedir_local
    return getData(num_classes, img, likelihood, gt)


def reconstruct_uncertainty_heatmap(datadir, unc_pred_mu, unc_pred_logvar, img_shape, unc_gt,prefix):
    eps = 0.1
    logfile = os.path.join(datadir, prefix+"_structure_info.txt")
    if not os.path.exists(os.path.dirname(logfile)):
        os.makedirs(os.path.dirname(logfile))

    manifold_filepath = os.path.join(datadir, "output/dimo_manifold.txt")
    vert_filepath = os.path.join(datadir, "output/dimo_vert.txt")

    unc_pred_mu = np.array(unc_pred_mu) 
    unc_pred_logvar = np.array(unc_pred_logvar)
    
    unc_pred_epistemic = np.var(unc_pred_mu, axis=0)
    unc_pred_aleatoric = np.exp(np.mean(unc_pred_logvar, axis=0))
    unc_pred_avg = np.mean(unc_pred_mu, axis=0)

    assert np.squeeze(unc_gt).shape == np.squeeze(unc_pred_avg).shape
    assert np.squeeze(unc_pred_aleatoric).shape == np.squeeze(unc_pred_avg).shape
    assert np.squeeze(unc_pred_epistemic).shape == np.squeeze(unc_pred_avg).shape

    if unc_gt.shape[0] == 1:
        unc_pred_aleatoric = np.reshape(unc_pred_aleatoric, 1)
        unc_pred_epistemic = np.reshape(unc_pred_epistemic, 1)
        unc_pred_avg = np.reshape(unc_pred_avg, 1)

    vert_info = np.loadtxt(vert_filepath)
    mini_image = np.zeros(img_shape)
    full_image = np.zeros(img_shape)

    writefile = open(logfile, 'a')

    manifold_cnt = -1
    with open(manifold_filepath, 'r') as manifold_info:
        reader = csv.reader(manifold_info, delimiter=' ')
        for row in reader:
            if len(row) != 3:
                if mini_image.sum() != 0:
                    manifold_cnt += 1

                    # write structure info to file
                    writestr = str(unc_pred_aleatoric[manifold_cnt]) + ',' + str(unc_pred_epistemic[manifold_cnt]) + ',' + str(unc_pred_avg[manifold_cnt]) + ',' + str(unc_gt[manifold_cnt]) + '\n'
                    writefile.write(writestr)
                    
                    if unc_pred_avg[manifold_cnt] >= 0.5:
                        full_image += mini_image

                mini_image = np.zeros(img_shape)
                continue

            v0 = int(row[0])
            v1 = int(row[1])

            mini_image[int(vert_info[v0,0]), int(vert_info[v0,1])] = eps + np.abs(1. - unc_gt[manifold_cnt+1])
            mini_image[int(vert_info[v1,0]), int(vert_info[v1,1])] = eps + np.abs(1. - unc_gt[manifold_cnt+1])

    assert manifold_cnt+1 == unc_pred_avg.shape[0]

    writefile.close()
    return full_image



if __name__ == "__main__":

    savedir = "/scr/saumgupta/crf-dmt/testing-temp/data/DRIVE/unet/test-outputs/trial34-mse-dmt-convmlp-Th002/convmlp-imgbatch/03_test/20/blur01_iid001"
    np_path = "/scr/saumgupta/crf-dmt/testing-temp/data/DRIVE/unet/test-outputs/trial7-ce-nosigmoid/epoch742-npy-dmtedges/03_test/20_384_437_/likelihood_almost01range.npy"
    imgpath = "/scr/saumgupta/crf-dmt/testing-temp/data/DRIVE/test/images/03_test.tif"

    img = Image.open(imgpath)
    img = transforms.ToTensor()(img)
    for j in range(img.shape[0]):
        meanval = img[j].mean()
        stdval = img[j].std()
        img[j] = (img[j] - meanval) / stdval

    img = torch.unsqueeze(img,0)[:,:,0:128,256:384]

    np_pred = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(np.load(np_path)),0),0)

    mydict = {}
    mydict['output_folder'] = savedir
    if not os.path.exists(mydict['output_folder']):
        os.makedirs(mydict['output_folder'])

    if not os.path.exists(os.path.join(mydict['output_folder'],'inputs')):
        shutil.copytree('inputs/', os.path.join(mydict['output_folder'],'inputs'))
    
    if not os.path.exists(os.path.join(mydict['output_folder'],'output')):
        os.makedirs(os.path.join(mydict['output_folder'], 'output'))

    getData_train(1,savedir, img, np_pred, None, 0.02, True)