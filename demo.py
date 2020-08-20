from rt_utils import *
from objects import *
from random_utils import *
from camera import Camera
import matplotlib.pyplot as plt
import time

# define constants
SAMPLES_PER_RAY = 100
MAX_RAY_BOUNCE = 50
MAX_RAY_LENGTH = 1e20


def create_random_scene(N):
    scene = [Sphere(np.float32([0, -1000, 0]), 1000, 
                    albedo=np.float32([0.5, 0.5, 0.5]),
                    material=Material.LAMBERTIAN)]

    for a in range(-N, N):
        for b in range(-N, N):
            choose_mat = np.random.random()
            center = np.float32([a+0.9*np.random.random(), 0.2, b+0.9*np.random.random()])

            if np.linalg.norm(center-np.float32([4, 0.2, 0])) > 0.9:
                if choose_mat < 0.8:
                    # diffuse
                    albedo = np.float32(np.random.random(3)**2)
                    scene += [Sphere(center, 0.2, 
                                     albedo=albedo, 
                                     material=Material.LAMBERTIAN)]
                elif choose_mat < 0.95:
                    # metal
                    albedo = np.float32(np.random.random(3)*0.5+0.5)
                    roughness = np.random.random()*0.5
                    scene += [Sphere(center, 0.2, 
                                     albedo=albedo, 
                                     material=Material.METAL,
                                     roughness=roughness)]
                else:
                    # glass
                    scene += [Sphere(center, 0.2, 
                                     material=Material.DIELECTRICS,
                                     ref_idx=1.5)]

    scene += [Sphere(np.float32([0, 1, 0]), 1.0, 
                     material=Material.DIELECTRICS,
                     ref_idx=1.5)]

    scene += [Sphere(np.float32([-4, 1, 0]), 1.0,
                     albedo=np.float32([0.4, 0.2, 0.1]), 
                     material=Material.LAMBERTIAN)]

    scene += [Sphere(np.float32([4, 1, 0]), 1.0,
                     albedo=np.float32([0.7, 0.6, 0.5]), 
                     material=Material.METAL,
                     roughness=0.)]

    return scene


if __name__ == '__main__':
    # define camera parameters
    img_wh, fov = (1200, 675), np.pi/6
    lookfrom = np.float32([13, 2, 3])
    lookat = np.float32([0, 0, -1])
    vup = np.float32([0, 1, 0])
    focus_dist = 10
    aperture = 0.1
    camera = Camera(lookfrom, lookat, vup, img_wh, fov, focus_dist, aperture)

    # cuda block size
    tpb = 32*16
    blocks = 64*64
    chunk = int(2**20)


    tscene = time.time()
    # construct the scene
    scene = create_random_scene(11)

    # construct scene buffers
    centers = []
    radii = []
    albedos = []
    materials = []
    roughnesses = []
    ref_idxs = []
    obj_idxs = []
    for obj_idx, obj in enumerate(scene):
        centers += [obj.center]
        radii += [obj.radius]
        albedos += [obj.albedo]
        materials += [obj.material.value]
        roughnesses += [obj.roughness]
        ref_idxs += [obj.ref_idx]
        obj_idxs += [obj_idx]
    centers = np.ascontiguousarray(centers, dtype=np.float32)
    radii = np.ascontiguousarray(radii, dtype=np.float32)
    albedos = np.ascontiguousarray(albedos, dtype=np.float32)
    materials = np.ascontiguousarray(materials, dtype=np.int32)
    roughnesses = np.ascontiguousarray(roughnesses, dtype=np.float32)
    ref_idxs = np.ascontiguousarray(ref_idxs, dtype=np.float32)
    obj_idxs = np.ascontiguousarray(obj_idxs, dtype=np.int32)

    print(f'scene has {len(scene)} objs, takes {time.time()-tscene:.4f} s to construct')
    
    
    # create random vectors
    rand_vec3 = random_unit_vector(int(1e6))

    # start ray tracing
    start_time = time.time()

    # create initial rays
    rays_o, rays_d = camera.get_rays(SAMPLES_PER_RAY)
    rays_o = np.ascontiguousarray(rays_o, dtype=np.float32)
    rays_d = np.ascontiguousarray(rays_d, dtype=np.float32)
    rays_idx = np.arange(len(rays_o))
    rays_color = np.ones_like(rays_o)

    bounce = 0
    while len(rays_o) > 1:
        N = len(rays_o)
        hit_record = np.zeros((N, 16), dtype=np.float32)
        hit_record[:, 0] += MAX_RAY_LENGTH # index 0 stores hit t
        hit_record[:, 1] += -1 # index 1 stores hit object
        # index 2~4 stores hit position
        # index 5~7 stores hit normal (normalized vec3)
        hit_record[:, 8] += 1 # index 8 stores if front face or not
        # index 9~11 stores scatter direction
        # index 12~14 stores albedo
        # index 15 stores scatter or not

        rand_vec3_ = rand_vec3[np.random.randint(rand_vec3.shape[0], size=len(rays_o)), :]
        
        for i in range(0, N, chunk): # process rays by chunk to avoid cuda OOM
            ray_sphere_intersect[blocks, tpb] \
                (rays_o[i:i+chunk], rays_d[i:i+chunk], 1e-4, 
                 centers, radii, albedos, materials, 
                 roughnesses, ref_idxs,
                 obj_idxs, hit_record[i:i+chunk],
                 rand_vec3_[i:i+chunk])

        # find hit or nohit
        rays_valid = hit_record[:, 0]<MAX_RAY_LENGTH
        rays_idx_hit = rays_idx[rays_valid] # (N) idx to assign to rays_color
        rays_idx_nohit = rays_idx[~rays_valid] # (N) idx to assign to rays_color

        # if ray doesn't hit anything, background color
        t = normalize(rays_d[~rays_valid])[:, 1:2]*0.5+0.5
        rays_color[rays_idx_nohit] *= (1.0-t)*np.array([1.0, 1.0, 1.0]) + t*np.array([0.5, 0.7, 1.0])

        bounce += 1
        if bounce >= MAX_RAY_BOUNCE:
            rays_color[rays_idx_hit] = 0.0 # black for rays that didn't stop
            break

        # if ray hits something, rays bounces to another direction to get its color
        # prepare rays for the next bounce
        hit_record_ = hit_record[rays_valid]
        # find which rays are scattered
        scatter = hit_record_[:, 15]>0

        # if it didn't scatter (the ray is absorbed), black
        rays_idx_nohit = rays_idx_hit[~scatter]
        rays_color[rays_idx_nohit] = 0

        # otherwise, continue tracing the ray
        hit_record_ = hit_record_[scatter]
        rays_o = np.ascontiguousarray(hit_record_[:, 2:5])
        rays_d = np.ascontiguousarray(hit_record_[:, 9:12])
        attenuation = hit_record_[:, 12:15]

        rays_idx_hit = rays_idx_hit[scatter]
        rays_color[rays_idx_hit] *= attenuation

        rays_idx = rays_idx_hit


    # finally, average the color
    rays_color = rays_color.reshape(-1, SAMPLES_PER_RAY, 3).mean(1)
    # gamma correction
    GAMMA = 2
    rays_color = rays_color**(1/GAMMA)

    end_time = time.time()

    print(f'rendering time: {end_time-start_time:.4f} s')
    print('saving to test.png ...')
    plt.imsave('test.png', rays_color.reshape(img_wh[1], img_wh[0], 3))