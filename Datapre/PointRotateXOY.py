# !/usr/bin/python
# -*- coding: UTF-8 -*-
"""
管段位姿调整
通过三个点确定一个平面，求其法向量
计算法向量在yoz平面上的投影向量与z轴的夹角，将法向量绕x轴旋转该角度，至xoz平面
计算旋转后的法向量与z轴的夹角，将该法向量旋转此角度则与z轴共线
同理，此三点所在的平面旋转这两个角度后即与z轴垂直
"""
import numpy as np
from matplotlib import pyplot as plt
from numpy import sin, cos, pi, mean
from numpy import arctan,linalg as la

from ThreePoint2Circle import points2circle


def RotXOY(pSta,pEnd,p_all):

    nP1P0=pEnd[1]-pEnd[0]#计算平面内的向量
    nP2P1=pEnd[2]-pEnd[1]
    a = np.array([[nP1P0[1],nP1P0[2]],[nP2P1[1],nP2P1[2]]])
    b = np.array([-100*nP1P0[0],-100*nP2P1[0]])
    c = np.linalg.solve(a, b)
    c=np.insert(c,0,100)
    '''
    c=[0,0 ,0]
    nP1P0 = pEnd[1] - pEnd[0]  # 计算平面内的向量
    nP2P1 = pEnd[2] - pEnd[1]
    c[0]=nP1P0[1]*nP2P1[2] -nP2P1[1] *nP1P0[2]
    c[1]=nP1P0[2]*nP2P1[0] -nP2P1[2] *nP1P0[0]
    c[2]=nP1P0[0]*nP2P1[1] -nP2P1[0] *nP1P0[1]
     '''
    #所有节点绕x轴旋转
    thex=-arctan(c[1]/c[2])
    Rx=[[1,0,0],
        [0,cos(thex),sin(thex)],
        [0,-sin(thex),cos(thex)]]
    Rot_pallx=[]
    for i in range(p_all.shape[0]):
        Rot_pallx.append(np.matmul(Rx, p_all[i]))
    Rot_pallx=np.array(Rot_pallx)
    #print('Rot_px',Rot_pallx)
    Rot_cx=np.matmul(Rx, c)

    # 所有节点绕y轴旋转
    they = arctan(Rot_cx[0]/ Rot_cx[2])
    Ry = [[cos(they),0,-sin(they)],
          [0,1,0],
          [sin(they), 0,cos(they)]]
    Rot_pall=[]
    for j in range(p_all.shape[0]):
        Rot_pall.append(np.matmul(Ry, Rot_pallx[j]))
    Rot_pall=np.array(Rot_pall)

    # 末节三点旋转及旋转后的中心点
    Rot_pEndx = []
    for i in range(pEnd.shape[0]):
        Rot_pEndx.append(np.matmul(Rx, pEnd[i]))
    Rot_pEndx = np.array(Rot_pEndx)
    Rot_pEnd = []
    for j in range(pEnd.shape[0]):
        Rot_pEnd.append(np.matmul(Ry, Rot_pEndx[j]))
    Rot_pEnd  = np.array(Rot_pEnd)
    cen_EndPi = points2circle(Rot_pEnd [0], Rot_pEnd[1], Rot_pEnd[2])

    #首节中心点旋转平移
    Rot_pStax=np.array(np.matmul(Rx, pSta))
    Rot_pSta=np.array(np.matmul(Ry, Rot_pStax))
    Rot_pSta =Rot_pSta-cen_EndPi

    Rot_pall=Rot_pall-cen_EndPi
    Rot_pall = np.array(Rot_pall)

    Rx180=[[1,0,0],
            [0,-1,0],
            [0,0,-1]]

    cenPall = np.mean(Rot_pall, axis=0)
    if cenPall[2]<0:
        Rot_pallz = []
        for i in range(Rot_pall.shape[0]):
            Rot_pallz.append(np.matmul(Rx180, Rot_pall[i]))
        Rot_pall=np.array(Rot_pallz)
        Rot_pSta=np.matmul(Rx180, Rot_pSta)

    thez=-arctan(Rot_pSta[0]/Rot_pSta[1])
    Rz=[[cos(thez),sin(thez),0],
        [-sin(thez), cos(thez),0],
        [0, 0, 1]]
    Rot_p=[]
    for i in range(Rot_pall.shape[0]):
        Rot_p.append(np.matmul(Rz,Rot_pall[i]))
    Rot_p = np.array(Rot_p)


    Rz180 = [[-1, 0, 0],
             [0, -1, 0],
             [0, 0, 1]]
    cenPall2 = np.mean(Rot_p, axis=0)
    #print(cenPall2)
    Rot_PP=[]
    if cenPall2[1]<0:
        for i in range(Rot_p.shape[0]):
            Rot_PP.append(np.matmul(Rz180, Rot_p[i]))
        Rot_PP=np.array(Rot_PP)
    else:
        Rot_PP=Rot_p

    return Rot_PP

if __name__=='__main__':
    cen_StaPi=np.array([357.64289403,378.49443676,954.87466678])
    Endthr_P=np.array([[278.454926,324.75589,1023.67908],[267.127838,339.709747,1017.73645],[270.701294, 350.74115, 1029.06665]])
    p_all=np.array([[19760.,        363.753296 ,  387.288727 , 965.371521],
                    [19761.,368.303436,377.323181,964.223572],
                    [19764.,         356.908783 ,  392.163116 ,  961.539978]])
    print(p_all)
    xoy= RotXOY(cen_StaPi,Endthr_P,p_all[:,1:])

    print(xoy)
    row = np.expand_dims(p_all[:, 0], axis=-1)
    print(row)
    p_rot_proc=np.append(xoy,row,axis=1)
    print(p_rot_proc)