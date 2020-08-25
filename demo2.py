from rt_utils import *
from objects import *
from random_utils import *
from background import *
from camera import Camera
from scipy.spatial.transform.rotation import Rotation as R
import matplotlib.pyplot as plt
import time

# define constants
SAMPLES_PER_RAY = 100
MAX_RAY_BOUNCE = 20
chunk = int(2**20)
GAMMA = 2.2


if __name__ == '__main__':
    tscene = time.time()
    # construct the scene
    mesh = TriangleMesh('/home/ubuntu/Downloads/Volkswagen/model/Touareg2.obj')
    mesh.vertices /= 1000
    scene = []
    scene += [mesh]
    # scene += [Sphere(np.float32([0, -1000.07, 0]), 1000, 
    #                  albedo=np.float32([0.5, 0.5, 0.5]),
    #                  material=Material.METAL,
    #                  roughness=0.1)]

    # scene += [Sphere(np.float32([0, 0.25, 3]), 0.25, 
    #                  material=Material.DIELECTRICS,
    #                  ref_idx=1.5)]

    scene += [Sphere(np.float32([-3, 1.2, -4]), 1.2,
                     albedo=np.float32([107, 62, 203])/255.0, 
                     material=Material.METAL,
                     roughness=0.)]

    # scene += [Sphere(np.float32([4, 1, 0]), 1.0,
    #                  albedo=np.float32([0.7, 0.6, 0.5]), 
    #                  material=Material.METAL,
    #                  roughness=0.)]
    # scene += [Sphere(np.float32([-2, 0.2, 5]), 0.2,
    #                  albedo=np.float32([178, 16, 45])/255.0, 
    #                  material=Material.LAMBERTIAN)]
    # scene += [Sphere(np.float32([5, 1.5, -15]), 1.5,
    #                  albedo=np.float32([34, 240, 175])/255.0, 
    #                  material=Material.LAMBERTIAN)]
    # scene += [Sphere(np.float32([-4, 0.4, 1]), 0.3,
    #                  albedo=np.float32([251, 95, 16])/255.0, 
    #                  material=Material.LAMBERTIAN)]
    
    # construct scene buffers
    # spheres
    centers = []
    radii = []
    albedos = []
    materials = []
    roughnesses = []
    ref_idxs = []
    obj_idxs = []

    # triangle meshes
    tris = []
    vert0 = []
    d1 = []
    d2 = []
    tri_norm_by_v = []
    tri_col = []
    bboxes = []

    for obj_idx, obj in enumerate(scene):
        if isinstance(obj, Sphere):
            centers += [obj.center]
            radii += [obj.radius]
            albedos += [obj.albedo]
            materials += [obj.material.value]
            roughnesses += [obj.roughness]
            ref_idxs += [obj.ref_idx]
            obj_idxs += [obj_idx]
        elif isinstance(obj, TriangleMesh):
            tris_ = mesh.vertices[mesh.triangles]
            vert0 += [tris_[:, 0]]
            d1 += [tris_[:, 1]-tris_[:, 0]]
            d2 += [tris_[:, 2]-tris_[:, 0]]
            tris += [tris_]
            tri_norm_by_v += [mesh.triangle_normals_by_vertex]
            tri_col += [mesh.triangle_colors]
            # only allow 1 bbox now. TODO: one bbox for each mesh
            bboxes = np.zeros(6, dtype=np.float32)
            bboxes[:3] = mesh.vertices.min(0)
            bboxes[3:] = mesh.vertices.max(0)

    if centers:
        centers = np.ascontiguousarray(centers, dtype=np.float32)
        radii = np.ascontiguousarray(radii, dtype=np.float32)
        albedos = np.ascontiguousarray(albedos, dtype=np.float32)
        materials = np.ascontiguousarray(materials, dtype=np.int32)
        roughnesses = np.ascontiguousarray(roughnesses, dtype=np.float32)
        ref_idxs = np.ascontiguousarray(ref_idxs, dtype=np.float32)
        obj_idxs = np.ascontiguousarray(obj_idxs, dtype=np.int32)

    if tris:
        tris = np.ascontiguousarray(np.vstack(tris), dtype=np.float32)
        vert0 = np.ascontiguousarray(np.vstack(vert0), dtype=np.float32)
        d1 = np.ascontiguousarray(np.vstack(d1), dtype=np.float32)
        d2 = np.ascontiguousarray(np.vstack(d2), dtype=np.float32)
        tri_norm_by_v = np.ascontiguousarray(np.vstack(tri_norm_by_v), dtype=np.float32)
        tri_col = np.ascontiguousarray(np.vstack(tri_col), dtype=np.float32)

    print(f'scene has {len(scene)} objs, takes {time.time()-tscene:.4f} s to construct')

    # construct lights (only 1 allowed currently, TODO: support multi light shadow)
    lights = np.float32([
                         [-6, 20, 10, 1, 1, 1],
                        #  [10, 20, 3, 1, 1, 1]
                        ])
    lights = lights.reshape(-1, 6)
    lights[:, :3] = normalize(lights[:, :3])
    L = len(lights)

    # construct bakcground
    bg = Skybox(['backgrounds/daiba/py.png', 
                 'backgrounds/daiba/nx.png', 
                 'backgrounds/daiba/pz.png', 
                 'backgrounds/daiba/px.png', 
                 'backgrounds/daiba/nz.png', 
                 'backgrounds/daiba/ny.png'])
    # bg = Skybox(['backgrounds/space1/top.png', 
    #              'backgrounds/space1/left.png', 
    #              'backgrounds/space1/back.png', 
    #              'backgrounds/space1/right.png', 
    #              'backgrounds/space1/front.png', 
    #              'backgrounds/space1/bottom.png'])
    
    # create random vectors
    rand_vec3 = random_unit_vector(int(1e6))

    # define camera parameters
    img_wh, fov = (480, 270), np.pi/6
    lookfrom = np.float32([-6, 3, 15])
    lookat = np.float32([0, 0.5, 0])
    vup = np.float32([0, 1, 0])

    for image_idx in range(1):
        # rotation = R.from_rotvec(-vup*np.pi/18)
        # lookfrom = rotation.apply(lookfrom)
        focus_dist = 15
        aperture = 0.2
        camera = Camera(lookfrom, lookat, vup, img_wh, fov, focus_dist, aperture)

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
            print(f'bounce {bounce}, {N} rays remaining ...')
            hit_record = HitRecord.get(N, 16)

            if len(centers)>0:
                blockdim = 32*16
                griddim = 32*32
                rand_vec3_ = rand_vec3[np.random.randint(rand_vec3.shape[0], size=N), :]
                for i in range(0, N, chunk): # process rays by chunk to avoid cuda OOM
                    ray_sphere_intersect[griddim, blockdim] \
                        (rays_o[i:i+chunk], rays_d[i:i+chunk], 1e-4, 0,
                        centers, radii, albedos, materials, 
                        roughnesses, ref_idxs,
                        obj_idxs,
                        lights, 
                        hit_record[i:i+chunk],
                        rand_vec3_[i:i+chunk])
            
            if len(tris)>0:
                blockdim = (32, 16)
                griddim = (32, 32)
                for i in range(0, N, chunk): # process rays by chunk to avoid cuda OOM
                    hit_record_ = np.zeros(min(N-i, chunk), dtype=np.float32)
                    ray_box_intersect[blockdim] \
                        (rays_o[i:i+chunk], rays_d[i:i+chunk], 
                        bboxes[:3], bboxes[3:], hit_record_)
                    chunk_idx = np.arange(i, min(i+chunk, N))
                    chunk_hit_idx = chunk_idx[hit_record_==1]
                    hit_record_ = hit_record[chunk_hit_idx]
                    ray_triangle_intersect[griddim, blockdim] \
                        (rays_o[chunk_hit_idx], rays_d[chunk_hit_idx], 1e-4, 0,
                        vert0, d1, d2, tri_norm_by_v, tri_col, 
                        lights,
                        hit_record_)
                    hit_record[chunk_hit_idx] = hit_record_

            # find hit or nohit
            rays_valid = hit_record[:, 0]<MAX_RAY_LENGTH
            rays_idx_hit = rays_idx[rays_valid] # (N) idx to assign to rays_color

            # if ray doesn't hit anything, background color
            # t = normalize(rays_d[~rays_valid])[:, 1:2]*0.5+0.5
            # rays_color[rays_idx[~rays_valid]] *= (1.0-t)*np.array([1.0, 1.0, 1.0]) + t*np.array([0.5, 0.7, 1.0])
            if N-len(rays_idx_hit)>0:
                rays_color[rays_idx[~rays_valid]] *= bg.get_colors(rays_d[~rays_valid])

            # if ray hits something, rays bounces to another direction to get its color
            # prepare rays for the next bounce
            hit_record = hit_record[rays_valid]
            # find which rays are scattered
            scatter = hit_record[:, 15]>0
            hit_record = hit_record[scatter]

            # if it didn't scatter (the ray is absorbed), black
            rays_color[rays_idx_hit[~scatter]] = 0

            # for rays that hit something and scattered, check if it can see the light
            ############################################################
            N = len(hit_record)
            if N <= 1:
                break

            rays_o = np.ascontiguousarray(hit_record[:, 2:5])
            rays_in_shadow = np.zeros(N)
            for l in range(L):
                hit_record_ = HitRecord.get(N, 1)
                rays_d = np.ascontiguousarray(np.tile(lights[l], (N, 1)))

                if len(centers)>0:
                    blockdim = 32*16
                    griddim = 32*32
                    rand_vec3_ = rand_vec3[np.random.randint(rand_vec3.shape[0], size=N), :]
                    for i in range(0, N, chunk): # process rays by chunk to avoid cuda OOM
                        ray_sphere_intersect[griddim, blockdim] \
                            (rays_o[i:i+chunk], rays_d[i:i+chunk], 1e-4, 1,
                            centers, radii, albedos, materials, 
                            roughnesses, ref_idxs,
                            obj_idxs,
                            lights, 
                            hit_record_[i:i+chunk],
                            rand_vec3_[i:i+chunk])
                
                if len(tris)>0:
                    blockdim = (32, 16)
                    griddim = (32, 32)
                    for i in range(0, N, chunk): # process rays by chunk to avoid cuda OOM
                        hit_record__ = np.zeros(min(N-i, chunk), dtype=np.float32)
                        ray_box_intersect[blockdim] \
                            (rays_o[i:i+chunk], rays_d[i:i+chunk], 
                            bboxes[:3], bboxes[3:], hit_record__)
                        chunk_idx = np.arange(i, min(i+chunk, N))
                        chunk_hit_idx = chunk_idx[hit_record__==1]
                        hit_record__ = hit_record_[chunk_hit_idx]
                        ray_triangle_intersect[griddim, blockdim] \
                            (rays_o[chunk_hit_idx], rays_d[chunk_hit_idx], 1e-4, 1,
                            vert0, d1, d2, tri_norm_by_v, tri_col, 
                            lights,
                            hit_record__)
                        hit_record_[chunk_hit_idx] = hit_record__

                rays_in_shadow += hit_record_[:, 0]<MAX_RAY_LENGTH
            ##############################################################

            rays_idx_hit = rays_idx_hit[scatter]

            # compute the attenuation color
            attenuation = hit_record[:, 12:15]
            attenuation *= (1-rays_in_shadow[:, np.newaxis]/L)
            rays_color[rays_idx_hit] *= attenuation

            # otherwise, continue tracing the ray
            rays_not_in_shadow = rays_in_shadow<L
            rays_idx_hit = rays_idx_hit[rays_not_in_shadow]
            rays_o = np.ascontiguousarray(rays_o[rays_not_in_shadow])
            rays_d = np.ascontiguousarray(hit_record[rays_not_in_shadow, 9:12])

            rays_idx = rays_idx_hit

            bounce += 1
            if bounce >= MAX_RAY_BOUNCE:
                break

            # # save the result of each step
            # rays_color_ = rays_color.reshape(-1, SAMPLES_PER_RAY, 3).mean(1)
            # rays_color_ = rays_color_**(1/GAMMA)
            # # clip to 0..1 range
            # rays_color_ = np.clip(rays_color_, 0, 1)
            # plt.imsave(f'bounce{bounce:03d}.png', rays_color_.reshape(img_wh[1], img_wh[0], 3))

        # save the final image
        rays_color = rays_color.reshape(-1, SAMPLES_PER_RAY, 3).mean(1)
        # rays_color = rays_color / (rays_color+1)
        # gamma correction
        rays_color = rays_color**(1/GAMMA)
        # # clip to 0..1 range
        rays_color = np.clip(rays_color, 0, 1)
        end_time = time.time()
            
        print(f'rendering time: {end_time-start_time:.4f} s')
        print('saving to test.png ...')
        plt.imsave(f'test{image_idx:03d}.png', rays_color.reshape(img_wh[1], img_wh[0], 3))