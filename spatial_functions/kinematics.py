import numpy as np
import modern_robotics as mr

def rotation_matrix(w, theta):
    R = (
        np.eye(3)
        + mr.VecToso3(w) * np.sin(theta)
        + (1 - np.cos(theta)) * mr.VecToso3(w) @ mr.VecToso3(w)
    )
    return R


def homogeneous_matrix(w, v_unit, theta):
    p = (
        np.eye(3) * theta
        + (1 - np.cos(theta)) * mr.VecToso3(w)
        + (theta - np.sin(theta)) * mr.VecToso3(w) @ mr.VecToso3(w)
    ) @ v_unit
    R = rotation_matrix(w, theta)
    H = mr.RpToTrans(R, p)
    return H
