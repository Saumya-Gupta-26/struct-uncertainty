'''
this uses the dimo_manifold.txt file to extract edges
saves all edge pieces and union

"binary_dots" option : adds dots for endpoints of each structure (only for "binary" option). we will overlay this on both input likelihood image and 'full' image [so for color dots, save as 3-channel; duplicate gray scale values across channels]
red dot = source [1,0,0 = RGB]
blue dot = destination [0,0,1 = RGB]
also gives green midpoint

ctrl+F nx to see where it is hardcoded



CUDA_VISIBLE_DEVICES=1 python3 dmt_extract_edges_5.py

'''

import subprocess

import sys
import numpy as np
import os
import time
import torch
import csv
from PIL import Image

t0 = time.time()

DIPHA_CONST = 8067171840
DIPHA_IMAGE_TYPE_CONST = 1
DIM = 3

class DMT:

    '''
        lh_map : .npy , HW and approx [0,1] range

        Creates dimo_manifold.txt
    '''
    def __init__(self, lh_map, Th=0.05):

        self.save_dir = "/data/saumgupta/miccai-tutorial"
        self.dipha_output_filename = os.path.join(self.save_dir,'inputs/complex.bin')
        self.vert_filename = os.path.join(self.save_dir,'inputs/vert.txt')
        self.dipha_edge_filename = os.path.join(self.save_dir,'inputs/dipha.edges')
        self.dipha_edge_txt = os.path.join(self.save_dir,'inputs/dipha_edges.txt')
        self.manifold_filepath = os.path.join(self.save_dir,"output/dimo_manifold.txt")
        self.vert_filepath = os.path.join(self.save_dir,"output/dimo_vert.txt")

        self.Th = Th * 255.
        self.lh_map = self.clip(self.sigmoid(lh_map))
        
        self.nx  = self.lh_map.shape[0]
        self.ny = self.lh_map.shape[1]
        self.nz = 1

        self.dmt_2d()



    def interpolate(self, nparr):
        omin = 0.0
        omax = 1.0
        imin  = np.min(nparr)
        imax = np.max(nparr)

        return (nparr-imin)*(omax-omin)/(imax-imin) + omin

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def clip(self, x):
        return np.clip(x, 0., 1.)
    
    def showLikelihood(self):
        return np.squeeze((np.clip(self.lh_map,0,1)*255.).astype(np.uint8))

    def dmt_2d(self):
        im_cube = np.zeros([self.nx, self.ny, self.nz])
        im_cube[:, :, 0] = self.lh_map

        with open(self.dipha_output_filename, 'wb') as output_file:
            np.int64(DIPHA_CONST).tofile(output_file)
            np.int64(DIPHA_IMAGE_TYPE_CONST).tofile(output_file)
            np.int64(self.nx * self.ny * self.nz).tofile(output_file)
            np.int64(DIM).tofile(output_file)
            np.int64(self.nx).tofile(output_file)
            np.int64(self.ny).tofile(output_file)
            np.int64(self.nz).tofile(output_file)
            for k in range(self.nz):
                sys.stdout.flush()
                for j in range(self.ny):
                    for i in range(self.nx):
                        val = int(-im_cube[i, j, k]*255)
                        np.float64(val).tofile(output_file)
            output_file.close()

        with open(self.vert_filename, 'w') as vert_file:
            for k in range(self.nz):
                sys.stdout.flush()
                for j in range(self.ny):
                    for i in range(self.nx):
                        vert_file.write(str(i) + ' ' + str(j) + ' ' + str(k) + ' ' + str(int(-im_cube[i, j, k] * 255)) + '\n')
            vert_file.close()

        '''
        mpiexec -n 32 ./dipha  path/to/input.bin path/to/output.bin path/to/output_for_morse.bin nx ny nz

        mpiexec -n 32 is only required if running with multiple (in this case 32) processes. The rest of the command can be used to run on a single process.

        path/to/input.bin - path to dipha input file generated in step 1 := inputs/complex.bin 
        path/to/output.bin - path to traditional dipha output file - this file is not used by our pipeline := inputs/diagram.bin
        path/to/output_for_morse.bin - path to output file our pipeline uses, contains persistence information of edges := inputs/dipha.edges
        '''

        subprocess.call(["mpiexec", "-n", "1", str(os.path.join(self.save_dir,"dipha-graph-recon/build/dipha")), str(os.path.join(self.save_dir,"inputs/complex.bin")), str(os.path.join(self.save_dir,"inputs/diagram.bin")), str(os.path.join(self.save_dir,"inputs/dipha.edges")), str(self.nx), str(self.ny), str(self.nz)])

        subprocess.call([str(os.path.join(self.save_dir,"src/loop.out")), str(self.dipha_edge_filename), str(self.dipha_edge_txt)])

        subprocess.call([str(os.path.join(self.save_dir,"src/manifold.out")), str(self.vert_filename), str(self.dipha_edge_txt), str(self.Th), str(os.path.join(self.save_dir,"output/"))])

    def showCriticalPoints(self): # only red and blue; using clubbed manifold code
        color_scale = 255.
        redcol = [1,0,0]
        greencol = [0,1,0]
        bluecol = [0,0,1]
        
        vert_info = np.loadtxt(self.vert_filepath)
        full_image = np.zeros([self.nx, self.ny, 3])
        full_image_nodot = np.zeros([self.nx, self.ny, 3])
        full_image_midpoint = np.zeros([self.nx, self.ny, 3])
        mini_image = np.zeros([self.nx, self.ny, 3])

        likelihood_dotoverlay = np.zeros([self.nx, self.ny, 3])
        likelihood_dotoverlay[:,:,0] = likelihood_dotoverlay[:,:,1] = likelihood_dotoverlay[:,:,2] = self.lh_map
        likelihood_nodot = np.copy(likelihood_dotoverlay)

        manifold_cnt = 0
        flag_red = None
        v0_blue = None 
        v1_blue = None

        v0_path = []
        v1_path = []

        with open(self.manifold_filepath, 'r') as manifold_info:
            reader = csv.reader(manifold_info, delimiter=' ')
            for row in reader:
                if len(row) != 3:
                    if mini_image.sum() != 0:
                        manifold_cnt += 1

                        full_image += mini_image
                        full_image_nodot += mini_image
                        full_image_midpoint += mini_image

                        # Setting colors -- 
                        #print("\nReds: {}, {}\nBlues: {}, {}\n".format(flag_red[0], flag_red[1], v0_blue, v1_blue))

                        # Setting 2 blues and their corresponding reds (don't set red if corresponding blue doesn't exist because this red is actually maxima/minima/blue for some other red)
                        if v0_blue:
                            likelihood_dotoverlay[int(vert_info[flag_red[0],0]), int(vert_info[flag_red[0],1]), :] = redcol
                            full_image[int(vert_info[flag_red[0],0]), int(vert_info[flag_red[0],1]), :] = redcol
                            likelihood_dotoverlay[int(vert_info[v0_blue,0]), int(vert_info[v0_blue,1]), :] = bluecol
                            full_image[int(vert_info[v0_blue,0]), int(vert_info[v0_blue,1]), :] = bluecol

                            midpath = len(v0_path)//2
                            midpath = v0_path[midpath][1] # arbitrary taking v1 instead of v0 in path
                            full_image_midpoint[int(vert_info[midpath,0]), int(vert_info[midpath,1]), :] = greencol

                        if v1_blue:
                            likelihood_dotoverlay[int(vert_info[flag_red[1],0]), int(vert_info[flag_red[1],1]), :] = redcol
                            full_image[int(vert_info[flag_red[1],0]), int(vert_info[flag_red[1],1]), :] = redcol
                            likelihood_dotoverlay[int(vert_info[v1_blue,0]), int(vert_info[v1_blue,1]), :] = bluecol
                            full_image[int(vert_info[v1_blue,0]), int(vert_info[v1_blue,1]), :] = bluecol

                            midpath = len(v1_path)//2
                            midpath = v1_path[midpath][1] # arbitrary taking v1 instead of v0 in path
                            full_image_midpoint[int(vert_info[midpath,0]), int(vert_info[midpath,1]), :] = greencol

                        #img_dmt = Image.fromarray(np.squeeze((mini_image*color_scale).astype(np.uint8)))
                        img_dmt = np.squeeze((mini_image*color_scale).astype(np.uint8))
                        #img_dmt.save(os.path.join(self.save_dir, "{}.png".format(manifold_cnt)))

                    mini_image = np.zeros([self.nx, self.ny, 3])
                    flag_red = None
                    v0_blue = None 
                    v1_blue = None
                    v0_path = []
                    v1_path = []
                    continue

                v0 = int(row[0])
                v1 = int(row[1])

                mini_image[int(vert_info[v0,0]), int(vert_info[v0,1]), :] = 1
                mini_image[int(vert_info[v1,0]), int(vert_info[v1,1]), :] = 1

                if flag_red is None:
                    flag_red = [v0,v1]

                else: # manifold starting from endpoints
                    if v0 == flag_red[0]:
                        v0_blue = v1 
                        v0_path.append([v0,v1])
                    elif v0 == flag_red[1]:
                        v1_blue = v1
                        v1_path.append([v0,v1]) 
                    else: # within the path
                        if v1_blue is None:
                            v0_blue = v1 
                            v0_path.append([v0,v1])
                        else:
                            v1_blue = v1
                            v1_path.append([v0,v1])


            #img_dmt = Image.fromarray(np.squeeze((np.clip(full_image,0,1)*color_scale).astype(np.uint8)))
            #img_dmt.save(os.path.join(self.save_dir, "full.png"))

            #img_dmt = Image.fromarray(np.squeeze((np.clip(full_image_nodot,0,1)*color_scale).astype(np.uint8)))
            #img_dmt.save(os.path.join(self.save_dir, "full_nodot.png"))

            #img_dmt = Image.fromarray(np.squeeze((np.clip(full_image_midpoint,0,1)*color_scale).astype(np.uint8)))
            #img_dmt.save(os.path.join(self.save_dir, "full_midpoint.png"))

            #img_dmt = Image.fromarray(np.squeeze((np.clip(likelihood_dotoverlay,0,1)*color_scale).astype(np.uint8)))
            img_dmt = np.squeeze((np.clip(likelihood_dotoverlay,0,1)*color_scale).astype(np.uint8))
            #img_dmt.save(os.path.join(self.save_dir, "likelihood_dotoverlay.png"))

            #img_dmt = Image.fromarray(np.squeeze((np.clip(likelihood_nodot,0,1)*color_scale).astype(np.uint8)))
            #img_dmt.save(os.path.join(self.save_dir, "likelihood_nodot.png"))

        print("Total images saved: {}".format(manifold_cnt+1))
        print("Total number of pixels in full image: {}".format(full_image_nodot.sum()))

        return img_dmt

    def showSkeleton(self):
        color_scale = 255.
        redcol = [1,0,0]
        greencol = [0,1,0]
        bluecol = [0,0,1]
        
        vert_info = np.loadtxt(self.vert_filepath)
        full_image = np.zeros([self.nx, self.ny, 3])
        full_image_nodot = np.zeros([self.nx, self.ny, 3])
        full_image_midpoint = np.zeros([self.nx, self.ny, 3])
        mini_image = np.zeros([self.nx, self.ny, 3])

        likelihood_dotoverlay = np.zeros([self.nx, self.ny, 3])
        likelihood_dotoverlay[:,:,0] = likelihood_dotoverlay[:,:,1] = likelihood_dotoverlay[:,:,2] = self.lh_map
        likelihood_nodot = np.copy(likelihood_dotoverlay)

        manifold_cnt = 0
        flag_red = None
        v0_blue = None 
        v1_blue = None

        v0_path = []
        v1_path = []

        with open(self.manifold_filepath, 'r') as manifold_info:
            reader = csv.reader(manifold_info, delimiter=' ')
            for row in reader:
                if len(row) != 3:
                    if mini_image.sum() != 0:
                        manifold_cnt += 1

                        full_image += mini_image
                        full_image_nodot += mini_image
                        full_image_midpoint += mini_image

                        # Setting colors -- 
                        #print("\nReds: {}, {}\nBlues: {}, {}\n".format(flag_red[0], flag_red[1], v0_blue, v1_blue))

                        # Setting 2 blues and their corresponding reds (don't set red if corresponding blue doesn't exist because this red is actually maxima/minima/blue for some other red)
                        if v0_blue:
                            likelihood_dotoverlay[int(vert_info[flag_red[0],0]), int(vert_info[flag_red[0],1]), :] = redcol
                            full_image[int(vert_info[flag_red[0],0]), int(vert_info[flag_red[0],1]), :] = redcol
                            likelihood_dotoverlay[int(vert_info[v0_blue,0]), int(vert_info[v0_blue,1]), :] = bluecol
                            full_image[int(vert_info[v0_blue,0]), int(vert_info[v0_blue,1]), :] = bluecol

                            midpath = len(v0_path)//2
                            midpath = v0_path[midpath][1] # arbitrary taking v1 instead of v0 in path
                            full_image_midpoint[int(vert_info[midpath,0]), int(vert_info[midpath,1]), :] = greencol

                        if v1_blue:
                            likelihood_dotoverlay[int(vert_info[flag_red[1],0]), int(vert_info[flag_red[1],1]), :] = redcol
                            full_image[int(vert_info[flag_red[1],0]), int(vert_info[flag_red[1],1]), :] = redcol
                            likelihood_dotoverlay[int(vert_info[v1_blue,0]), int(vert_info[v1_blue,1]), :] = bluecol
                            full_image[int(vert_info[v1_blue,0]), int(vert_info[v1_blue,1]), :] = bluecol

                            midpath = len(v1_path)//2
                            midpath = v1_path[midpath][1] # arbitrary taking v1 instead of v0 in path
                            full_image_midpoint[int(vert_info[midpath,0]), int(vert_info[midpath,1]), :] = greencol

                        #img_dmt = Image.fromarray(np.squeeze((mini_image*color_scale).astype(np.uint8)))
                        img_dmt = np.squeeze((mini_image*color_scale).astype(np.uint8))
                        #img_dmt.save(os.path.join(self.save_dir, "{}.png".format(manifold_cnt)))

                    mini_image = np.zeros([self.nx, self.ny, 3])
                    flag_red = None
                    v0_blue = None 
                    v1_blue = None
                    v0_path = []
                    v1_path = []
                    continue

                v0 = int(row[0])
                v1 = int(row[1])

                mini_image[int(vert_info[v0,0]), int(vert_info[v0,1]), :] = 1
                mini_image[int(vert_info[v1,0]), int(vert_info[v1,1]), :] = 1

                if flag_red is None:
                    flag_red = [v0,v1]

                else: # manifold starting from endpoints
                    if v0 == flag_red[0]:
                        v0_blue = v1 
                        v0_path.append([v0,v1])
                    elif v0 == flag_red[1]:
                        v1_blue = v1
                        v1_path.append([v0,v1]) 
                    else: # within the path
                        if v1_blue is None:
                            v0_blue = v1 
                            v0_path.append([v0,v1])
                        else:
                            v1_blue = v1
                            v1_path.append([v0,v1])

            img_dmt = np.squeeze((np.clip(full_image_nodot,0,1)*color_scale).astype(np.uint8))
            #img_dmt.save(os.path.join(self.save_dir, "full_nodot.png"))

        print("Total images saved: {}".format(manifold_cnt+1))
        print("Total number of pixels in full image: {}".format(full_image_nodot.sum()))

        return img_dmt

    def showManifolds(self):
        color_scale = 255.
        redcol = [1,0,0]
        greencol = [0,1,0]
        bluecol = [0,0,1]
        
        vert_info = np.loadtxt(self.vert_filepath)
        full_image = np.zeros([self.nx, self.ny, 3])
        full_image_nodot = np.zeros([self.nx, self.ny, 3])
        full_image_coloredges = np.zeros([self.nx, self.ny, 3])
        mini_image = np.zeros([self.nx, self.ny, 3])

        likelihood_dotoverlay = np.zeros([self.nx, self.ny, 3])
        likelihood_dotoverlay[:,:,0] = likelihood_dotoverlay[:,:,1] = likelihood_dotoverlay[:,:,2] = self.lh_map
        likelihood_nodot = np.copy(likelihood_dotoverlay)

        manifold_cnt = 0
        flag_red = None
        v0_blue = None 
        v1_blue = None

        v0_path = []
        v1_path = []

        with open(self.manifold_filepath, 'r') as manifold_info:
            reader = csv.reader(manifold_info, delimiter=' ')
            for row in reader:
                if len(row) != 3:
                    if mini_image.sum() != 0:
                        manifold_cnt += 1

                        full_image += mini_image
                        full_image_nodot += mini_image

                        mini_image = mini_image * np.random.rand(3)
                        full_image_coloredges += mini_image

                        # Setting colors -- 
                        #print("\nReds: {}, {}\nBlues: {}, {}\n".format(flag_red[0], flag_red[1], v0_blue, v1_blue))

                        # Setting 2 blues and their corresponding reds (don't set red if corresponding blue doesn't exist because this red is actually maxima/minima/blue for some other red)
                        if v0_blue:
                            likelihood_dotoverlay[int(vert_info[flag_red[0],0]), int(vert_info[flag_red[0],1]), :] = redcol
                            full_image[int(vert_info[flag_red[0],0]), int(vert_info[flag_red[0],1]), :] = redcol
                            likelihood_dotoverlay[int(vert_info[v0_blue,0]), int(vert_info[v0_blue,1]), :] = bluecol
                            full_image[int(vert_info[v0_blue,0]), int(vert_info[v0_blue,1]), :] = bluecol

                            midpath = len(v0_path)//2
                            midpath = v0_path[midpath][1] # arbitrary taking v1 instead of v0 in path

                        if v1_blue:
                            likelihood_dotoverlay[int(vert_info[flag_red[1],0]), int(vert_info[flag_red[1],1]), :] = redcol
                            full_image[int(vert_info[flag_red[1],0]), int(vert_info[flag_red[1],1]), :] = redcol
                            likelihood_dotoverlay[int(vert_info[v1_blue,0]), int(vert_info[v1_blue,1]), :] = bluecol
                            full_image[int(vert_info[v1_blue,0]), int(vert_info[v1_blue,1]), :] = bluecol

                            midpath = len(v1_path)//2
                            midpath = v1_path[midpath][1] # arbitrary taking v1 instead of v0 in path

                        #img_dmt = Image.fromarray(np.squeeze((mini_image*color_scale).astype(np.uint8)))
                        img_dmt = np.squeeze((mini_image*color_scale).astype(np.uint8))
                        #img_dmt.save(os.path.join(self.save_dir, "{}.png".format(manifold_cnt)))

                    mini_image = np.zeros([self.nx, self.ny, 3])
                    flag_red = None
                    v0_blue = None 
                    v1_blue = None
                    v0_path = []
                    v1_path = []
                    continue

                v0 = int(row[0])
                v1 = int(row[1])

                mini_image[int(vert_info[v0,0]), int(vert_info[v0,1]), :] = 1
                mini_image[int(vert_info[v1,0]), int(vert_info[v1,1]), :] = 1

                if flag_red is None:
                    flag_red = [v0,v1]

                else: # manifold starting from endpoints
                    if v0 == flag_red[0]:
                        v0_blue = v1 
                        v0_path.append([v0,v1])
                    elif v0 == flag_red[1]:
                        v1_blue = v1
                        v1_path.append([v0,v1]) 
                    else: # within the path
                        if v1_blue is None:
                            v0_blue = v1 
                            v0_path.append([v0,v1])
                        else:
                            v1_blue = v1
                            v1_path.append([v0,v1])

            img_dmt = np.squeeze((np.clip(full_image_coloredges,0,1)*color_scale).astype(np.uint8))
            #img_dmt.save(os.path.join(self.save_dir, "full_midpoint.png"))

        print("Total images saved: {}".format(manifold_cnt+1))
        print("Total number of pixels in full image: {}".format(full_image_nodot.sum()))

        return img_dmt




def save_edges_binary():
    nx = ny = 128
    manifold_filepath = "output/dimo_manifold.txt"
    vert_filepath = "output/dimo_vert.txt"

    vert_info = np.loadtxt(vert_filepath)
    full_image = np.zeros([nx, ny])
    mini_image = np.zeros([nx, ny])

    manifold_cnt = 0
    with open(manifold_filepath, 'r') as manifold_info:
        reader = csv.reader(manifold_info, delimiter=' ')
        for row in reader:
            if len(row) != 3:
                if mini_image.sum() != 0:
                    manifold_cnt += 1
                    full_image += mini_image
                    img_dmt = Image.fromarray(np.squeeze((mini_image*255.).astype(np.uint8)))
                    img_dmt.save(os.path.join(save_dir, "{}.png".format(manifold_cnt)))
                    mini_image = np.zeros([nx, ny])
                continue

            v0 = int(row[0])
            v1 = int(row[1])

            mini_image[int(vert_info[v0,0]), int(vert_info[v0,1])] = 1
            mini_image[int(vert_info[v1,0]), int(vert_info[v1,1])] = 1

        img_dmt = Image.fromarray(np.squeeze((np.clip(full_image,0,1)*255.).astype(np.uint8)))
        img_dmt.save(os.path.join(save_dir, "full.png"))

    print("Total images saved: {}".format(manifold_cnt+1))
    print("Total number of pixels in full image: {}".format(full_image.sum()))








def save_edges_likelihood(likelihood_map):
    nx = ny = 128
    manifold_filepath = "output/dimo_manifold.txt"
    vert_filepath = "output/dimo_vert.txt"

    vert_info = np.loadtxt(vert_filepath)
    full_mask = np.zeros([nx, ny])
    mini_mask = np.zeros([nx, ny])

    manifold_cnt = 0
    with open(manifold_filepath, 'r') as manifold_info:
        reader = csv.reader(manifold_info, delimiter=' ')
        for row in reader:
            if len(row) != 3:
                if mini_mask.sum() != 0:
                    manifold_cnt += 1
                    full_mask += mini_mask
                    mini_image = mini_mask * likelihood_map
                    img_dmt = Image.fromarray(np.squeeze((mini_image*255.).astype(np.uint8)))
                    img_dmt.save(os.path.join(save_dir, "{}.png".format(manifold_cnt)))
                    mini_mask = np.zeros([nx, ny])
                continue

            v0 = int(row[0])
            v1 = int(row[1])

            mini_mask[int(vert_info[v0,0]), int(vert_info[v0,1])] = 1
            mini_mask[int(vert_info[v1,0]), int(vert_info[v1,1])] = 1

        full_image = full_mask * likelihood_map
        img_dmt = Image.fromarray(np.squeeze((np.clip(full_image,0,1)*255.).astype(np.uint8)))
        img_dmt.save(os.path.join(save_dir, "full.png"))

    print("Total images saved: {}".format(manifold_cnt+1))
    print("Total number of pixels in full image: {}".format(full_mask.sum()))



def save_edges(likelihood_map):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if save_type == "binary":
        save_edges_binary()
    elif save_type == "binary_dots":
        save_edges_binary_dots_v1(likelihood_map)
    elif save_type == "likelihood":
        save_edges_likelihood(likelihood_map)
    elif save_type == "persistence":
        save_edges_persistence()




if __name__ == "__main__":

    np_pred = clip(sigmoid(np.load(np_path)))

    thresh_op = np.where(np_pred >= 0.5, 1., 0.)
    img_dmt = Image.fromarray(np.squeeze((np.clip(thresh_op,0,1)*255.).astype(np.uint8)))
    img_dmt.save(os.path.join(save_dir, "thresholded.png"))

    np_pred = np.expand_dims(np.expand_dims(np_pred, axis=0), axis=0) #NCHW
    y_pred = torch.from_numpy(np_pred).cuda()
    print(y_pred.size())

    dmt(y_pred,global_th) # dimo_manifold.txt created

    save_edges(np.squeeze(np_pred))

    # save edges as separate images; value of pixels in image can be either 1) one (so a binary img); 2) value from likelihood ; 3) persistence value

    print("Time taken: {}".format(time.time() - t0))