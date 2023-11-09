import math
from scipy.spatial.transform import Rotation
import torch
from mpl_toolkits.mplot3d import Axes3D
from numpy import cos, sin, pi, mean, sqrt
from scipy.optimize import root, fsolve
import numpy as np
from matplotlib import pyplot as plt

def initial_Theoretical_Point(R,fv,rD,m):

    """
    arguments:
        R: the bending radius
        rD: the outer radius
        fv: the twisting angle
    returns:
        Points: the outer points after the cross-section distortion
    """
    R=R/1000
    rD/=1000

    P1=[]

    f0=0.0
    angle1 = np.linspace(0, (pi - fv) / 2, num=1000)
    for a in angle1:
        def func(P):
            return ((R + (rD - P * cos(a)) * cos(f0 + fv)) * ((rD - P * cos(a)) ** 2 + (P * sin(a)) ** 2) ** 0.5 *cos(fv) / (R * rD) - 1)
        P1.append(float(fsolve(func, 0.0)))
    P1 = np.array(P1)
    disp_max=max(P1)
    #椭圆化方程计算离散点（半周椭圆化）
    a = rD-disp_max
    b = rD
    angle1 = np.linspace(pi/2.0, 3*pi/2.0, num=19)
    #angle10 = angle1/ pi * 180
    #print(angle10)
    ellipse_point=[]
    for t in angle1:
        x =  a * math.cos(t)
        y =  b * math.sin(t)
        ellipse_point.append([x,y,0])
    angle2 = np.linspace(-pi/2.0+pi/18.0, pi/2.0-pi/18.0, num=17)
    #angle20=angle2/pi*180
    #print(angle20)
    for t in angle2:
        x =  rD * math.cos(t)
        y =  rD * math.sin(t)
        ellipse_point.append([x,y,0])
    ellipse_point=np.array(ellipse_point)

    rot = Rotation.from_euler('zyx', [fv,0,0]).as_matrix()
    ellipse_point = (rot @ ellipse_point.T ).T
    #print('ellipse_point',ellipse_point.shape)
    return ellipse_point


def BatchCrossPoi(p):
    Cross = []
    for i in range(p.shape[0]):
        Crossi=initial_Theoretical_Point(p[i][0],p[i][1],p[i][2],p[i][3])
        Cross.append(Crossi)
    Cross=np.array(Cross)
    Cross=torch.from_numpy(Cross)
    return Cross
if __name__ == '__main__':
    p=np.array([[500,0.1,15,70],[500,0.1,15,70],[500,0.1,15,70],[500,0.1,15,70]])
    Point=BatchCrossPoi(p)

    fig = plt.figure()
    ax = Axes3D(fig)
    fig.add_axes(ax)
    plt.xlabel("X")
    plt.ylabel("Y")
    print('Point',Point.shape)

    for i in range(len(Point[0])):
        x = Point[0][i][0]
        y = Point[0][i][1]
        z = Point[0][i][2]
        ax.scatter(x, y, z, color="red")
    plt.show()

