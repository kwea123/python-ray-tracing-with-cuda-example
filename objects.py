import open3d as o3d
import numpy as np
from enum import Enum


class Material(Enum):
    LAMBERTIAN = 0
    METAL = 1
    DIELECTRICS = 2


MAX_RAY_LENGTH = 1e20

class HitRecord:
    """
    # index 0 stores hit t
    # index 1 stores hit object
    # index 2~4 stores hit position
    # index 5~7 stores hit normal (normalized vec3)
    # index 8 stores if front face or not
    # index 9~11 stores scatter direction
    # index 12~14 stores albedo
    # index 15 stores scatter or not
    """
    @staticmethod
    def get(N, f):
        hit_record = np.zeros((N, f), dtype=np.float32)
        hit_record[:, 0] = MAX_RAY_LENGTH
        if f > 1:
            hit_record[:, 1] = -1

        return hit_record


class Sphere:
    def __init__(self, center, radius, 
                 albedo=np.zeros(3, dtype=np.float32),
                 material=Material.LAMBERTIAN, # TODO: replace this with metallic parameter
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


class TriangleMesh:
    def __init__(self, filepath):
        self.mesh = o3d.io.read_triangle_mesh(filepath)

        self.vertices = np.asarray(self.mesh.vertices, dtype=np.float32)
        self.triangles = np.asarray(self.mesh.triangles)
        self.vertex_colors = np.asarray(self.mesh.vertex_colors, dtype=np.float32)
        self.triangle_colors = self.vertex_colors[self.triangles].mean(1)

        # self.mesh.compute_triangle_normals()
        # self.triangle_normals = np.asarray(self.mesh.triangle_normals, dtype=np.float32)

        self.mesh.compute_vertex_normals()
        self.vertex_normals = np.asarray(self.mesh.vertex_normals, dtype=np.float32)
        self.triangle_normals_by_vertex = self.vertex_normals[self.triangles] # (M, 3, 3)