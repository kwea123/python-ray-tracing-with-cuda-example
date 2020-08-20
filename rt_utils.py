import numpy as np
from numba import cuda


def normalize(v):
    return v/(np.linalg.norm(v, axis=1, keepdims=True)+1e-10)


@cuda.jit('void(f4[:,:], f4[:,:], f4, f4[:,:], f4[:], f4[:,:], i4[:], f4[:], f4[:], i4[:], f4[:,:], f4[:,:])', fastmath=True)
def ray_sphere_intersect(rays_o, rays_d, min_t, 
                         centers, radii, albedos, materials, roughnesses, ref_idxs,
                         obj_idxs, hit_record,
                         rand_vec3):
    """
    Computes if the rays intersect the spheres
    Ref: https://tavianator.com/2011/ray_box.html
    
    Inputs:
        rays_o: (N, 3) ray origins
        rays_d: (N, 3) ray directions (not necessarily normalized)
        min_t: float, a little amount to offset the rays
        centers: (M, 3) sphere centers
        radii: (M), sphere radii
        albedos: (M, 3)
        materials: (M), specified in objects.py Material class
        roughnesses: (M)
        ref_idxs: (M)
        obj_idxs: (M)
        hit_record: (N, ?)
        rand_vec3: (N, 3) random vec3
    """
    N, M = len(rays_o), len(centers)
    
    start_i = cuda.grid(1)
    grid_i = cuda.gridDim.x * cuda.blockDim.x
    
    for i in range(start_i, N, grid_i):
        o, d = rays_o[i], rays_d[i]
        for j in range(M):
            center = centers[j]
            radius = radii[j]
            albedo = albedos[j]
            material = materials[j]
            roughness = roughnesses[j]
            ref_idx = ref_idxs[j]
            obj_idx = obj_idxs[j]
            co0 = o[0]-center[0]
            co1 = o[1]-center[1]
            co2 = o[2]-center[2]
            a = d[0]**2+d[1]**2+d[2]**2
            half_b = co0*d[0]+co1*d[1]+co2*d[2]
            c = co0**2+co1**2+co2**2-radius**2
            disc = half_b**2-a*c
            if disc>0:
                t = (-half_b-disc**0.5) / a
                if t>min_t and t<hit_record[i, 0]:
                    hit_record[i, 0] = t
                    hit_record[i, 1] = obj_idx
                    hit_record[i, 2] = o[0]+t*d[0]
                    hit_record[i, 3] = o[1]+t*d[1]
                    hit_record[i, 4] = o[2]+t*d[2]
                    hit_record[i, 12] = albedo[0]
                    hit_record[i, 13] = albedo[1]
                    hit_record[i, 14] = albedo[2]
                    r_inv = radius**-1
                    n0 = (co0+t*d[0])*r_inv
                    n1 = (co1+t*d[1])*r_inv
                    n2 = (co2+t*d[2])*r_inv
                    ddotn = d[0]*n0+d[1]*n1+d[2]*n2
                    if ddotn < 0: # front face
                        hit_record[i, 5] = n0
                        hit_record[i, 6] = n1
                        hit_record[i, 7] = n2
                        hit_record[i, 8] = 1
                    else:
                        hit_record[i, 5] = -n0
                        hit_record[i, 6] = -n1
                        hit_record[i, 7] = -n2
                        hit_record[i, 8] = 0

                    if material == 0: # lambertian
                        hit_record[i, 9] = hit_record[i, 5] + rand_vec3[i, 0]
                        hit_record[i, 10] = hit_record[i, 6] + rand_vec3[i, 1]
                        hit_record[i, 11] = hit_record[i, 7] + rand_vec3[i, 2]
                        hit_record[i, 15] = 1
                    elif material == 1: # metal
                        hit_record[i, 9] = d[0]-2*ddotn*n0 + roughness*rand_vec3[i, 0]
                        hit_record[i, 10] = d[1]-2*ddotn*n1 + roughness*rand_vec3[i, 1]
                        hit_record[i, 11] = d[2]-2*ddotn*n2 + roughness*rand_vec3[i, 2]
                        hit_record[i, 15] = \
                            hit_record[i, 9]*n0 + \
                            hit_record[i, 10]*n1 + \
                            hit_record[i, 11]*n2 > 0
                    elif material == 2: # dielectrics
                        etai_over_etat = 1/ref_idx if ddotn < 0 else ref_idx
                        d_norm = a**0.5
                        d_unit0 = d[0]/d_norm
                        d_unit1 = d[1]/d_norm
                        d_unit2 = d[2]/d_norm
                        cos_theta = min(-(d_unit0*n0+d_unit1*n1+d_unit2*n2), 1.0)
                        sin_theta = (1-cos_theta**2)**0.5
                        if etai_over_etat*sin_theta>1.0: # total reflection
                            hit_record[i, 9] = d[0]-2*ddotn*n0
                            hit_record[i, 10] = d[1]-2*ddotn*n1
                            hit_record[i, 11] = d[2]-2*ddotn*n2
                        # r0 = ((1-etai_over_etat)/(1+etai_over_etat))**2
                        # reflect_prob = r0+(1-r0)*(1-cos_theta)**5
                        # if abs(rand_vec3[i, 0]) < reflect_prob:
                        #     hit_record[i, 9] = d[0]-2*ddotn*n0
                        #     hit_record[i, 10] = d[1]-2*ddotn*n1
                        #     hit_record[i, 11] = d[2]-2*ddotn*n2
                        else: # refraction
                            r_out_perp0 = etai_over_etat*(d_unit0+cos_theta*n0)
                            r_out_perp1 = etai_over_etat*(d_unit1+cos_theta*n1)
                            r_out_perp2 = etai_over_etat*(d_unit2+cos_theta*n2)
                            f = -(1-r_out_perp0**2-r_out_perp1**2-r_out_perp2**2)**0.5
                            r_out_parallel0 = f*n0
                            r_out_parallel1 = f*n1
                            r_out_parallel2 = f*n2
                            hit_record[i, 9] = r_out_perp0+r_out_parallel0
                            hit_record[i, 10] = r_out_perp1+r_out_parallel1
                            hit_record[i, 11] = r_out_perp2+r_out_parallel2
                        hit_record[i, 15] = 1



# @cuda.jit('void(f4[:,:], f4[:,:], f4[:], f4[:], f4[:])', fastmath=True)
# def ray_box_intersect(rays_o, rays_d, xyz_min, xyz_max, hit):
#     """
#     Computes if the rays intersect the AABB
#     Ref: https://tavianator.com/2011/ray_box.html
    
#     Inputs:
#         rays_o: (N, 3) ray origins
#         rays_d: (N, 3) ray directions (not necessarily normalized)
#         xyz_min: (3) AABB min
#         xyz_max: (3) AABB max
#         hit: (N) 1 if ray intersects with the AABB, 0 otherwise
#     """
#     N = len(rays_o)
    
#     start_i = cuda.grid(1)
#     grid_i = cuda.gridDim.x * cuda.blockDim.x
    
#     for i in range(start_i, N, grid_i):
#         o, d = rays_o[i], rays_d[i]
#         d0_inv = 1/(d[0]+1e-10)
#         d1_inv = 1/(d[1]+1e-10)
#         d2_inv = 1/(d[2]+1e-10)
        
#         tx1 = (xyz_min[0]-o[0])*d0_inv
#         tx2 = (xyz_max[0]-o[0])*d0_inv
#         ty1 = (xyz_min[1]-o[1])*d1_inv
#         ty2 = (xyz_max[1]-o[1])*d1_inv
#         tz1 = (xyz_min[2]-o[2])*d2_inv
#         tz2 = (xyz_max[2]-o[2])*d2_inv
        
#         tmin = max(min(tx1, tx2), min(ty1, ty2), min(tz1, tz2))
#         tmax = min(max(tx1, tx2), max(ty1, ty2), max(tz1, tz2))
#         if tmax>=tmin:
#             hit[i] = 1


# @cuda.jit('void(f4[:,:], f4[:,:], f4[:,:], f4[:,:], f4[:,:], f4[:,:], f4[:,:], f4[:,:])', fastmath=True)
# def ray_triangle_intersect(rays_o, rays_d, v0, d1, d2, tri_col, rays_hit_t, rays_color):
#     """
#     Computes the intersections of rays and triangles using CUDA
#     Ref: https://blog.csdn.net/weixin_34288121/article/details/92181954
    
#     Inputs:
#         rays_o: (N, 3) ray origins
#         rays_d: (N, 3) ray directions (not necessarily normalized)
#         v0: (M, 3) coordinate for vertex 0 of the triangles
#         d1: (M, 3) v1-v0
#         d2: (M, 3) v2-v0
#         tri_col: (M, 3) triangle colors
#         rays_hit_t: (N, 1) the nearest hit parameter t
#         rays_color: (N, 3) the hit color
#     """
#     N, M = len(rays_o), len(d1)
    
#     start_i, start_j = cuda.grid(2)
#     grid_i = cuda.gridDim.x * cuda.blockDim.x
#     grid_j = cuda.gridDim.y * cuda.blockDim.y
    
#     for i in range(start_i, N, grid_i):
#         o, d = rays_o[i], rays_d[i]
#         hit_t = rays_hit_t[i, 0]
#         for j in range(start_j, M, grid_j):
#             p0, e1, e2 = v0[j], d1[j], d2[j]
#             q0 = d[1]*e2[2]-d[2]*e2[1]
#             q1 = d[2]*e2[0]-d[0]*e2[2]
#             q2 = d[0]*e2[1]-d[1]*e2[0]
#             a = e1[0]*q0+e1[1]*q1+e1[2]*q2
#             if abs(a)<1e-10: # ray almost parallel to the triangle
#                 continue
#             f = a**(-1) # MUCH FASTER than 1.0/a !!!
#             s0 = o[0]-p0[0]
#             s1 = o[1]-p0[1]
#             s2 = o[2]-p0[2]
#             u = f*(s0*q0+s1*q1+s2*q2)
#             if u<0 or u>1:
#                 continue
#             r0 = s1*e1[2]-s2*e1[1]
#             r1 = s2*e1[0]-s0*e1[2]
#             r2 = s0*e1[1]-s1*e1[0]
#             v = f*(d[0]*r0+d[1]*r1+d[2]*r2)
#             if v<0 or u+v>1:
#                 continue
#             t = f*(e2[0]*r0+e2[1]*r1+e2[2]*r2)
#             if t<0 or t>hit_t:
#                 continue
#             hit_t = t
#             old_hit_t = cuda.atomic.min(rays_hit_t, (i, 0), hit_t)
#             if hit_t <= old_hit_t:
#                 rays_color[i, 0] = tri_col[j, 0]
#                 rays_color[i, 1] = tri_col[j, 1]
#                 rays_color[i, 2] = tri_col[j, 2]