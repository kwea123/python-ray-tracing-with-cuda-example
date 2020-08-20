# import open3d as o3d
import numpy as np
from enum import Enum


class Material(Enum):
    LAMBERTIAN = 0
    METAL = 1
    DIELECTRICS = 2


class Sphere:
    def __init__(self, center, radius, 
                 albedo=np.zeros(3, dtype=np.float32),
                 material=Material.LAMBERTIAN,
                 roughness=0.0,
                 ref_idx=1.0):
        self.center = center
        self.radius = radius
        self.albedo = albedo
        self.material = material
        self.roughness = roughness
        self.ref_idx = ref_idx

        if material == Material.DIELECTRICS:
            self.albedo = np.ones(3, dtype=np.float32)


# class TriangleMesh:
#     def __init__(self, filepath):
#         self.mesh = o3d.io.read_triangle_mesh(filepath)