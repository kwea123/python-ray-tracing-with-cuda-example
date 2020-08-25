import numpy as np


def random_in_unit_disk(N):
    phi = 2*np.pi*np.random.random(N)
    r = np.random.random(N) ** 0.5
    x = r * np.sin(phi)
    y = r * np.cos(phi)
    z = np.zeros_like(x)
    res = np.stack((x, y, z), 1)
    return np.ascontiguousarray(res, dtype=np.float32)


# def random_in_unit_sphere(N):
#     """
#     Generate (N, 3) vectors in unit sphere
#     """
#     phi = 2*np.pi*np.random.random(N)
#     cos_theta = np.random.random(N)*2-1
#     u = np.random.random(N)
#     theta = np.arccos(cos_theta)
#     r = u**(1/3)
#     x = r * np.sin(theta) * np.cos(phi)
#     y = r * np.sin(theta) * np.sin(phi)
#     z = r * np.cos(theta)
#     return np.stack((x, y, z), 1).astype(np.float32)


def random_unit_vector(N):
    a = 2*np.pi*np.random.random(N)
    z = np.random.random(N)*2-1
    r = (1-z**2)**0.5
    res = np.stack((r*np.cos(a), r*np.sin(a), z), 1)
    return np.ascontiguousarray(res, dtype=np.float32)