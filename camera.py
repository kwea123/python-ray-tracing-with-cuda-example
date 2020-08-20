import numpy as np
from random_utils import *

def normalize(v):
    return v/(np.linalg.norm(v)+1e-10)


class Camera:
    def __init__(self, 
                 lookfrom=np.float32([0, 0, 0]), 
                 lookat=np.float32([0, 0, -1]),
                 vup=np.float32([0, 1, 0]),
                 img_wh=(480, 270),
                 fov=np.pi/2,
                 focus_dist=0.,
                 aperture=0.,
                 ):

        self.origin = lookfrom

        self.w = normalize(lookfrom-lookat)
        self.u = normalize(np.cross(vup, self.w))
        self.v = np.cross(self.w, self.u)
        if focus_dist <= 0:
            self.focus_dist = np.linalg.norm(lookfrom-lookat)
        else:
            self.focus_dist = focus_dist
        self.lens_radius = aperture/2

        self.W = img_wh[0]
        self.H = img_wh[1]
        self.focal = 0.5*self.W/np.tan(0.5*fov)

        self.i, self.j = np.mgrid[:self.H,:self.W][::-1]
        

    def get_rays(self, samples_per_ray=1):
        i = np.stack([self.i]*samples_per_ray, -1)
        j = np.stack([self.j]*samples_per_ray, -1)

        if samples_per_ray > 1:
            i = i + np.random.random(i.shape)
            j = j + np.random.random(j.shape)

        rays_d = (i[..., np.newaxis]-self.W/2)/self.focal*self.focus_dist*self.u + \
                 -(j[..., np.newaxis]-self.H/2)/self.focal*self.focus_dist*self.v + \
                 -self.focus_dist*self.w
        rays_d = rays_d.reshape(-1, 3).astype(np.float32)
        rays_o = np.tile(self.origin, (len(rays_d), 1))

        # compute ray offset for depth of field
        if self.lens_radius > 0:
            offset = self.lens_radius * random_in_unit_disk(len(rays_d)) # (N, 3)
            offset = offset[:, 0:1] * self.u + offset[:, 1:2] * self.v
            rays_o += offset
            rays_d -= offset

        return rays_o, rays_d