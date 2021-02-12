import numpy as np
import warnings

def set_alpha():
    return 2.

def u_cbf(x, u_candidate, clipper=10.):
    ran = _u_of_x(x, clipper)
    rmin, rmax = ran[0], ran[1]
    out = u_candidate
    if h(x) < 1e-1:
        if x[0] < 0:
            out = rmax
        else:
            out = rmin
    else:
        if u_candidate < rmin:
            out = rmin
        elif u_candidate > rmax:
            out = rmax
    return np.array([out])

def u_cbf_pure(x, u_candidate, clipper):
    ran = _u_of_x(x, clipper)
    rmin, rmax = ran[0], ran[1]
    out = u_candidate
    if False:
        if x[0] < 0:
            out = rmax
        else:
            out = rmin
    else:
        if u_candidate < rmin:
            out = rmin
        elif u_candidate > rmax:
            out = rmax
    return np.array([out])


def h(x, alpha=set_alpha()):
        return 1 - x[0]**2 - alpha*x[1]**2


def _u_of_x(x, clipper, k=2.6, alpha=set_alpha()):
    assert x.shape[0]==2, 'shape_error'
    al = alpha
    m = 1
    l = 1
    g = 10.
    gamma = pow(10, k)
    th, thd = x
    if thd == 0:
        p = abs(m*g*np.sin(th)/2)
        ran = [-clipper, -p] if th > 0 else [p, clipper]
    else:
        u_thres = np.roots([-(6*al*thd)/(m*l**2), -(2*th*thd + 3*al*g/l*thd*np.sin(th) - gamma*h(x, al))])[0]
        u_thres = np.clip(u_thres, -clipper, clipper)
        if thd < 0:
            u_thres = max(u_thres, -clipper)
            ran = [u_thres, clipper]
        else:
            u_thres = min(u_thres, clipper)
            ran = [-clipper, u_thres]
    return ran
