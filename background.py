import numpy as np
import cv2


class Skybox:
    def __init__(self, cubemap):
        if len(cubemap)==1:
            self.cubemap = cv2.imread(cubemap[0])[:,:,::-1]/255.0 # rgb
            self.size = self.cubemap.shape[0]//3
        elif len(cubemap)==6:
            self.top = cv2.imread(cubemap[0])[:,:,::-1]/255.0
            self.left = cv2.imread(cubemap[1])[:,:,::-1]/255.0
            self.back = cv2.imread(cubemap[2])[:,:,::-1]/255.0
            self.right = cv2.imread(cubemap[3])[:,:,::-1]/255.0
            self.front = cv2.imread(cubemap[4])[:,:,::-1]/255.0
            self.bottom = cv2.imread(cubemap[5])[:,:,::-1]/255.0
            
            self.size = self.top.shape[0]
            self.cubemap = np.zeros((self.size*3, self.size*4, 3), dtype=np.float32)
            self.cubemap[:self.size, self.size:2*self.size] = self.top
            self.cubemap[-self.size:, self.size:2*self.size] = self.bottom
            self.cubemap[self.size:2*self.size, :self.size] = self.left
            self.cubemap[self.size:2*self.size, self.size:2*self.size] = self.back
            self.cubemap[self.size:2*self.size, 2*self.size:3*self.size] = self.right
            self.cubemap[self.size:2*self.size, 3*self.size:] = self.front

    def get_colors(self, rays_d):
        """
        Inputs:
            rays_d: (N, 3)

        Outputs:
            colors: (N, 3)
        """
        rays_d_ = rays_d/np.max(np.abs(rays_d), axis=1, keepdims=True)
        hit_top = rays_d_[:, 1] == 1
        hit_bottom = rays_d_[:, 1] == -1
        hit_left = rays_d_[:, 0] == -1
        hit_back = rays_d_[:, 2] == -1
        hit_right = rays_d_[:, 0] == 1
        hit_front = rays_d_[:, 2] == 1

        u = np.select([hit_top, hit_bottom, hit_left, hit_back, hit_right, hit_front],
                      [(rays_d_[:, 0]+1)*1/8+1/4, (rays_d_[:, 0]+1)*1/8+1/4,
                       (-rays_d_[:, 2]+1)*1/8+0/4, (rays_d_[:, 0]+1)*1/8+1/4,
                       (rays_d_[:, 2]+1)*1/8+2/4, (-rays_d_[:, 0]+1)*1/8+3/4])

        v = np.select([hit_top, hit_bottom, hit_left, hit_back, hit_right, hit_front],
                      [(-rays_d_[:, 2]+1)*1/6+0/3, (rays_d_[:, 2]+1)*1/6+2/3,
                       (-rays_d_[:, 1]+1)*1/6+1/3, (-rays_d_[:, 1]+1)*1/6+1/3,
                       (-rays_d_[:, 1]+1)*1/6+1/3, (-rays_d_[:, 1]+1)*1/6+1/3])

        u = (u*4*self.size).astype(np.float32)
        v = (v*3*self.size).astype(np.float32)

        colors = []
        remap_chunk = int(3e4)
        for i in range(0, len(rays_d), remap_chunk):
            colors += [cv2.remap(self.cubemap, 
                                 u[i:i+remap_chunk],
                                 v[i:i+remap_chunk],
                                 interpolation=cv2.INTER_LINEAR)[:, 0]]
        colors = np.vstack(colors)

        return colors

        # return hit_top[:, np.newaxis] * np.float32([1, 0, 0]) + \
        #        hit_bottom[:, np.newaxis] * np.float32([1, 0, 0]) + \
        #        hit_left[:, np.newaxis] * np.float32([0, 0, 1])# + \
            #    hit_back[:, np.newaxis] * np.float32([0, 1, 0]) + \
            #    hit_right[:, np.newaxis] * np.float32([0, 0, 1]) + \
            #    hit_front[:, np.newaxis] * np.float32([0, 1, 0])
