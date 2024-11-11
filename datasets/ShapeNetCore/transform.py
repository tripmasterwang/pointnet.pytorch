import random
import numpy as np


class NormalizeCoord(object):
    def __call__(self, pcd):
    # BLANK
        return pcd


class CenterShift(object):
    def __init__(self, apply_z=True):
        self.apply_z = apply_z

    def __call__(self, pcd):
        x_min, y_min, z_min = pcd.min(axis=0)
        x_max, y_max, _ = pcd.max(axis=0)
        if self.apply_z:
            shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]
        else:
            shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, 0]
        pcd -= shift
        return pcd


class RandomRotate(object):
    def __init__(self, angle=None, center=None, axis="z", always_apply=False, p=0.5):
        self.angle = [-1, 1] if angle is None else angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, pcd):
        if random.random() > self.p:
            return pcd
        angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if self.center is None:
            x_min, y_min, z_min = pcd.min(axis=0)
            x_max, y_max, z_max = pcd.max(axis=0)
            center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
        else:
            center = self.center
        pcd -= center
        pcd = np.dot(pcd, np.transpose(rot_t))
        pcd += center
        return pcd


class Compose(object):
    def __init__(self, transforms= []):
        self.transforms = transforms

    def __call__(self, pcd):
        for transform in self.transforms:
            pcd = transform(pcd)
        return pcd
    
