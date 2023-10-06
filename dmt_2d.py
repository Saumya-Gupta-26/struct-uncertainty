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
        self.compute_features()



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

        subprocess.call(["mpiexec", "-n", "1", str(os.path.join(self.save_dir,"dipha-graph-recon/build/dipha")), str(os.path.join(self.save_dir,"inputs/complex.bin")), str(os.path.join(self.save_dir,"inputs/diagram.bin")), str(os.path.join(self.save_dir,"inputs/dipha.edges")), str(self.nx), str(self.ny), str(self.nz)])

        subprocess.call([str(os.path.join(self.save_dir,"src/loop.out")), str(self.dipha_edge_filename), str(self.dipha_edge_txt)])

        subprocess.call([str(os.path.join(self.save_dir,"src/manifold.out")), str(self.vert_filename), str(self.dipha_edge_txt), str(self.Th), str(os.path.join(self.save_dir,"output/"))])


    def compute_features(self):
        color_scale = 255.
        redcol = [1,0,0]
        bluecol = [0,0,1]
        
        vert_info = np.loadtxt(self.vert_filepath)
        full_image = np.zeros([self.nx, self.ny, 3])
        full_image_nodot = np.zeros([self.nx, self.ny, 3])
        full_image_coloredges = np.zeros([self.nx, self.ny, 3])
        mini_image = np.zeros([self.nx, self.ny, 3])

        likelihood_dotoverlay = np.zeros([self.nx, self.ny, 3])
        likelihood_dotoverlay[:,:,0] = likelihood_dotoverlay[:,:,1] = likelihood_dotoverlay[:,:,2] = self.lh_map

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


            self.critical_points = np.squeeze((np.clip(likelihood_dotoverlay,0,1)*color_scale).astype(np.uint8))
            self.skeleton = np.squeeze((np.clip(full_image_nodot,0,1)*color_scale).astype(np.uint8))
            self.manifolds = np.squeeze((np.clip(full_image_coloredges,0,1)*color_scale).astype(np.uint8))



    def showCriticalPoints(self): # only red and blue; using clubbed manifold code
        return self.critical_points

    def showSkeleton(self):
        return self.skeleton

    def showManifolds(self):
        return self.manifolds