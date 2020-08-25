import numpy as np
from numba import cuda

PI = np.pi


def normalize(v):
    return v/(np.linalg.norm(v, axis=1, keepdims=True)+1e-10)


@cuda.jit('void(f4[:,:], f4[:,:], f4, i4, f4[:,:], f4[:], f4[:,:], i4[:], f4[:], f4[:], i4[:], f4[:,:], f4[:,:], f4[:,:])', fastmath=True)
def ray_sphere_intersect(rays_o, rays_d, min_t, t_only,
                         centers, radii, albedos, materials, roughnesses, ref_idxs,
                         obj_idxs, lights, hit_record,
                         rand_vec3):
    """
    Computes if the rays intersect the spheres
    
    Inputs:
        rays_o: (N, 3) ray origins
        rays_d: (N, 3) ray directions (not necessarily normalized)
        min_t: float, a little amount to offset the rays
        t_only: only retrieve t or not
        centers: (M, 3) sphere centers
        radii: (M), sphere radii
        albedos: (M, 3)
        materials: (M), specified in objects.py Material class
        roughnesses: (M)
        ref_idxs: (M)
        obj_idxs: (M)
        lights: (L, 6) normalized direction, color
        hit_record: (N, ?) light directions (normalized)
        rand_vec3: (N, 3) random vec3
    """
    N, M, L = len(rays_o), len(centers), len(lights)
    
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
                    if t_only > 0:
                        continue
                    hit_record[i, 1] = obj_idx
                    hit_record[i, 2] = o[0]+t*d[0]
                    hit_record[i, 3] = o[1]+t*d[1]
                    hit_record[i, 4] = o[2]+t*d[2]

                    r_inv = radius**-1
                    n0 = (co0+t*d[0])*r_inv
                    n1 = (co1+t*d[1])*r_inv
                    n2 = (co2+t*d[2])*r_inv
                    ndotd = d[0]*n0+d[1]*n1+d[2]*n2
                    if ndotd < 0: # front face
                        hit_record[i, 8] = 1
                    else:
                        n0 *= -1
                        n1 *= -1
                        n2 *= -1
                        hit_record[i, 8] = 0
                    hit_record[i, 5] = n0
                    hit_record[i, 6] = n1
                    hit_record[i, 7] = n2

                    amb = 0.05
                    ka = 1
                    hit_record[i, 12] = ka*amb
                    hit_record[i, 13] = ka*amb
                    hit_record[i, 14] = ka*amb

                    if material == 0: # lambertian
                        hit_record[i, 9] = n0 + rand_vec3[i, 0]
                        hit_record[i, 10] = n1 + rand_vec3[i, 1]
                        hit_record[i, 11] = n2 + rand_vec3[i, 2]
                        hit_record[i, 12] += albedo[0]
                        hit_record[i, 13] += albedo[1]
                        hit_record[i, 14] += albedo[2]
                        hit_record[i, 15] = 1
                    elif material == 1: # metal
                        hit_record[i, 9] = d[0]-2*ndotd*n0 + roughness*rand_vec3[i, 0]
                        hit_record[i, 10] = d[1]-2*ndotd*n1 + roughness*rand_vec3[i, 1]
                        hit_record[i, 11] = d[2]-2*ndotd*n2 + roughness*rand_vec3[i, 2]
                        d_norm = (d[0]**2+d[1]**2+d[2]**2)**0.5
                        v0 = -d[0]/d_norm
                        v1 = -d[1]/d_norm
                        v2 = -d[2]/d_norm
                        kd, ks = 1, 1
                        for l in range(L):
                            light = lights[l]
                            ndotl = max(0, n0*light[0]+n1*light[1]+n2*light[2])
                            h0_ = light[0]+v0
                            h1_ = light[1]+v1
                            h2_ = light[2]+v2
                            h_norm = (h0_**2+h1_**2+h2_**2)**0.5
                            h0 = h0_/h_norm
                            h1 = h1_/h_norm
                            h2 = h2_/h_norm
                            spec = 32 # exponent
                            ndoth = max(0, n0*h0+n1*h1+n2*h2)**spec
                            hit_record[i, 12] += kd*ndotl*albedo[0]+ks*ndoth*light[3]
                            hit_record[i, 13] += kd*ndotl*albedo[1]+ks*ndoth*light[4]
                            hit_record[i, 14] += kd*ndotl*albedo[2]+ks*ndoth*light[5]
                        hit_record[i, 15] = \
                            hit_record[i, 9]*n0 + \
                            hit_record[i, 10]*n1 + \
                            hit_record[i, 11]*n2 > 0
                    elif material == 2: # dielectrics
                        etai_over_etat = 1/ref_idx if ndotd < 0 else ref_idx
                        d_norm = a**0.5
                        d_unit0 = d[0]/d_norm
                        d_unit1 = d[1]/d_norm
                        d_unit2 = d[2]/d_norm
                        cos_theta = min(-(d_unit0*n0+d_unit1*n1+d_unit2*n2), 1.0)
                        sin_theta = (1-cos_theta**2)**0.5
                        if etai_over_etat*sin_theta>1.0: # total reflection
                            hit_record[i, 9] = d[0]-2*ndotd*n0
                            hit_record[i, 10] = d[1]-2*ndotd*n1
                            hit_record[i, 11] = d[2]-2*ndotd*n2
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
                        hit_record[i, 12] += albedo[0]
                        hit_record[i, 13] += albedo[1]
                        hit_record[i, 14] += albedo[2]
                        hit_record[i, 15] = 1


@cuda.jit('void(f4[:,:], f4[:,:], f4[:], f4[:], f4[:])', fastmath=True)
def ray_box_intersect(rays_o, rays_d, xyz_min, xyz_max, hit):
    """
    Computes if the rays intersect the AABB
    Ref: https://tavianator.com/2011/ray_box.html
    
    Inputs:
        rays_o: (N, 3) ray origins
        rays_d: (N, 3) ray directions (not necessarily normalized)
        xyz_min: (3) AABB min
        xyz_max: (3) AABB max
        hit: (N) 1 if ray intersects with the AABB, 0 otherwise
    """
    N = len(rays_o)
    
    start_i = cuda.grid(1)
    grid_i = cuda.gridDim.x * cuda.blockDim.x
    
    for i in range(start_i, N, grid_i):
        o, d = rays_o[i], rays_d[i]
        d0_inv = 1/(d[0]+1e-10)
        d1_inv = 1/(d[1]+1e-10)
        d2_inv = 1/(d[2]+1e-10)
        
        tx1 = (xyz_min[0]-o[0])*d0_inv
        tx2 = (xyz_max[0]-o[0])*d0_inv
        ty1 = (xyz_min[1]-o[1])*d1_inv
        ty2 = (xyz_max[1]-o[1])*d1_inv
        tz1 = (xyz_min[2]-o[2])*d2_inv
        tz2 = (xyz_max[2]-o[2])*d2_inv
        
        tmin = max(min(tx1, tx2), min(ty1, ty2), min(tz1, tz2))
        tmax = min(max(tx1, tx2), max(ty1, ty2), max(tz1, tz2))
        if tmax>=tmin:
            hit[i] = 1


@cuda.jit('f4(f4, f4, f4, f4, f4, f4, f4)', device=True, inline=True, fastmath=True)
def DistributionGGX(n0, n1, n2, h0, h1, h2, roughness):
    a2 = roughness**4
    ndoth2 = max(n0*h0+n1*h1+n2*h2, 0)**2
	
    denom = ndoth2*(a2-1)+1
    denom = PI * denom**2
	
    return a2/denom


@cuda.jit('f4(f4, f4)', device=True, inline=True, fastmath=True)
def GeometrySchlickGGX(ndotv, roughness):
    r = roughness+1.0
    k = (r**2)/8

    num   = ndotv
    denom = ndotv * (1.0 - k) + k
	
    return num / denom


@cuda.jit('UniTuple(f4, 2)(f4, f4, f4, f4, f4, f4, f4, f4)', device=True, inline=True, fastmath=True)
def GeometrySmith(n0, n1, n2, v0, v1, v2, ndotl, roughness):
    ndotv = max(n0*v0+n1*v1+n2*v2, 0)
    ggx2  = GeometrySchlickGGX(ndotv, roughness)
    ggx1  = GeometrySchlickGGX(ndotl, roughness)
	
    return ggx1 * ggx2, ndotv


@cuda.jit('UniTuple(f4, 3)(f4, f4, f4, f4)', device=True, inline=True, fastmath=True)
def fresnelSchlick(cosTheta, F00, F01, F02):
    f = (1-cosTheta)**5
    return F00+(1-F00)*f, F01+(1-F01)*f, F02+(1-F02)*f


@cuda.jit('void(f4[:,:], f4[:,:], f4, i4, f4[:,:], f4[:,:], f4[:,:], f4[:,:,:], f4[:,:], f4[:,:], f4[:,:])', fastmath=True)
def ray_triangle_intersect(rays_o, rays_d, min_t, t_only,
                           vert0, d1, d2, 
                           tri_norm_by_v, tri_col, 
                           lights,
                           hit_record):
    """
    Computes the intersections of rays and triangles using CUDA
    Ref: https://blog.csdn.net/weixin_34288121/article/details/92181954
    
    Inputs:
        rays_o: (N, 3) ray origins
        rays_d: (N, 3) ray directions (not necessarily normalized)
        min_t: float, a little amount to offset the rays
        t_only: only retrieve t or not
        vert0: (M, 3) coordinate for vertex 0 of the triangles
        d1: (M, 3) vert1-vert0
        d2: (M, 3) vert2-vert0
        tri_norm_by_v: (M, 3, 3) triangle normals by vertex
        tri_col: (M, 3) triangle colors
        lights: (L, 3) light directions (normalized)
        hit_record: (N, ?)
    """
    N, M, L = len(rays_o), len(d1), len(lights)
    
    start_i, start_j = cuda.grid(2)
    grid_i = cuda.gridDim.x * cuda.blockDim.x
    grid_j = cuda.gridDim.y * cuda.blockDim.y
    
    for i in range(start_i, N, grid_i):
        o, d = rays_o[i], rays_d[i]
        for j in range(start_j, M, grid_j):
            p0, e1, e2 = vert0[j], d1[j], d2[j]
            q0 = d[1]*e2[2]-d[2]*e2[1]
            q1 = d[2]*e2[0]-d[0]*e2[2]
            q2 = d[0]*e2[1]-d[1]*e2[0]
            a = e1[0]*q0+e1[1]*q1+e1[2]*q2
            if abs(a)<1e-5: # ray almost parallel to the triangle
                continue
            f = a**(-1) # MUCH FASTER than 1.0/a !!!
            s0 = o[0]-p0[0]
            s1 = o[1]-p0[1]
            s2 = o[2]-p0[2]
            u = f*(s0*q0+s1*q1+s2*q2)
            if u<0 or u>1:
                continue
            r0 = s1*e1[2]-s2*e1[1]
            r1 = s2*e1[0]-s0*e1[2]
            r2 = s0*e1[1]-s1*e1[0]
            v = f*(d[0]*r0+d[1]*r1+d[2]*r2)
            if v<0 or u+v>1:
                continue
            t = f*(e2[0]*r0+e2[1]*r1+e2[2]*r2)
            if t<min_t or t>hit_record[i, 0]:
                continue
            old_hit_t = cuda.atomic.min(hit_record, (i, 0), t)
            if t_only>0 or old_hit_t<t:
                continue
            hit_record[i, 2] = o[0]+t*d[0]
            hit_record[i, 3] = o[1]+t*d[1]
            hit_record[i, 4] = o[2]+t*d[2]
            n0 = tri_norm_by_v[j, 0, 0]*(1-u-v)+tri_norm_by_v[j, 1, 0]*u+tri_norm_by_v[j, 2, 0]*v
            n1 = tri_norm_by_v[j, 0, 1]*(1-u-v)+tri_norm_by_v[j, 1, 1]*u+tri_norm_by_v[j, 2, 1]*v
            n2 = tri_norm_by_v[j, 0, 2]*(1-u-v)+tri_norm_by_v[j, 1, 2]*u+tri_norm_by_v[j, 2, 2]*v
            n_norm = (n0**2+n1**2+n2**2)**0.5
            n0/=n_norm
            n1/=n_norm
            n2/=n_norm
            ndotd = d[0]*n0+d[1]*n1+d[2]*n2
            if ndotd < 0: # front face
                hit_record[i, 8] = 1
            else:
                n0 *= -1
                n1 *= -1
                n2 *= -1
                hit_record[i, 8] = 0
            hit_record[i, 5] = n0
            hit_record[i, 6] = n1
            hit_record[i, 7] = n2

            # every face is metal
            hit_record[i, 9] = d[0]-2*ndotd*n0 #+ roughness*rand_vec3[i, 0]
            hit_record[i, 10] = d[1]-2*ndotd*n1 #+ roughness*rand_vec3[i, 1]
            hit_record[i, 11] = d[2]-2*ndotd*n2 #+ roughness*rand_vec3[i, 2]
            hit_record[i, 15] = \
                hit_record[i, 9]*n0 + \
                hit_record[i, 10]*n1 + \
                hit_record[i, 11]*n2 > 0

            d_norm = (d[0]**2+d[1]**2+d[2]**2)**0.5
            v0 = -d[0]/d_norm
            v1 = -d[1]/d_norm
            v2 = -d[2]/d_norm

            # Blinn-Phon shading
            amb = 0.03
            ka, kd, ks = 1, 1, 5
            hit_record[i, 12] = ka*amb
            hit_record[i, 13] = ka*amb
            hit_record[i, 14] = ka*amb
            for l in range(L):
                light = lights[l]
                ndotl = max(0, n0*light[0]+n1*light[1]+n2*light[2])
                h0_ = light[0]+v0
                h1_ = light[1]+v1
                h2_ = light[2]+v2
                h_norm = (h0_**2+h1_**2+h2_**2)**0.5
                h0 = h0_/h_norm
                h1 = h1_/h_norm
                h2 = h2_/h_norm
                spec = 32 # exponent
                ndoth = max(0, n0*h0+n1*h1+n2*h2)**spec
                hit_record[i, 12] += kd*ndotl*tri_col[j, 0]+ks*ndoth*light[3]
                hit_record[i, 13] += kd*ndotl*tri_col[j, 1]+ks*ndoth*light[4]
                hit_record[i, 14] += kd*ndotl*tri_col[j, 2]+ks*ndoth*light[5]

            # # PBR
            # metallic = 1.0
            # roughness = 0.1
            # F00 = 0.04*(1-metallic)+tri_col[j, 0]*metallic
            # F01 = 0.04*(1-metallic)+tri_col[j, 1]*metallic
            # F02 = 0.04*(1-metallic)+tri_col[j, 2]*metallic

            # # ambient color
            # # replace with environment map
            # hit_record[i, 12] = 0.03*tri_col[j, 0]
            # hit_record[i, 13] = 0.03*tri_col[j, 1]
            # hit_record[i, 14] = 0.03*tri_col[j, 2]

            # for l in range(L):
            #     light = lights[l]
            #     ndotl = max(0, n0*light[0]+n1*light[1]+n2*light[2])
            #     h0_ = light[0]+v0
            #     h1_ = light[1]+v1
            #     h2_ = light[2]+v2
            #     h_norm = (h0_**2+h1_**2+h2_**2)**0.5
            #     h0 = h0_/h_norm
            #     h1 = h1_/h_norm
            #     h2 = h2_/h_norm

            #     # directional light
            #     radiance0 = light[3]
            #     radiance1 = light[4]
            #     radiance2 = light[5]

            #     # cook-torrance brdf
            #     ndf = DistributionGGX(n0, n1, n2, h0, h1, h2, roughness)
            #     g, ndotv = GeometrySmith(n0, n1, n2, v0, v1, v2, ndotl, roughness)
            #     hdotv = max(h0*v0+h1*v1+h2*v2, 0)
            #     F0, F1, F2 = fresnelSchlick(hdotv, F00, F01, F02)

            #     ks0, ks1, ks2 = F0, F1, F2
            #     kd0, kd1, kd2 = 1-ks0, 1-ks1, 1-ks2
            #     kd0 *= 1-metallic
            #     kd1 *= 1-metallic	  
            #     kd2 *= 1-metallic	  
                
            #     sp0, sp1, sp2 = ndf*g*F0, ndf*g*F1, ndf*g*F2
            #     denominator = max(4*ndotv*ndotl, 1e-3)
            #     specular0 = sp0 / denominator
            #     specular1 = sp1 / denominator
            #     specular2 = sp2 / denominator

            #     hit_record[i, 12] += (kd0*tri_col[j, 0]/PI + specular0)*radiance0*ndotl
            #     hit_record[i, 13] += (kd1*tri_col[j, 1]/PI + specular1)*radiance1*ndotl
            #     hit_record[i, 14] += (kd2*tri_col[j, 2]/PI + specular2)*radiance2*ndotl
