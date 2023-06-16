from examples.courses.udes_gro640.prob.dura1101 import dh2T, dhs2T
import numpy as np


def test_dh2T():
    r = 7.0
    d = 5.0
    theta = np.pi/2
    alpha = -np.pi/2
    T = dh2T(r, d, theta, alpha)
    assert T.shape == (4, 4)
    r_ba = np.array([T[0, 3], T[1, 3], T[2, 3]])
    assert np.array_equal(r_ba, np.array([r*np.cos(theta), r*np.sin(theta), d]))

def test_dhs2T():
    r = np.array([7.0, 6.0, 5.0])
    d = np.array([5.0, 4.0, 3.0])
    theta = np.array([np.pi/2, np.pi/3, np.pi/4])
    alpha = np.array([-np.pi/2, -np.pi/3, -np.pi/4])
    T = dhs2T(r, d, theta, alpha)
    assert T.shape == (4, 4)
