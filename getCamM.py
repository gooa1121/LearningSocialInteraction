import math
import numpy as np
import matplotlib
matplotlib.use("nbagg")
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import copy

def getAnglesFromR(R):#R is from the extrinsic transform matrix
    cam_R = R.transpose() #find camera pose

    sy = math.sqrt(cam_R[2,1] * cam_R[2,1] +  cam_R[2,2] * cam_R[2,2])
    x = math.atan2(cam_R[2,1] , cam_R[2,2])
    y = math.atan2(-cam_R[2,0], sy)
    z = math.atan2(cam_R[1,0], cam_R[0,0])

    return np.array([x, y, z])

def calRz(rvec):
    rotation = getAnglesFromR(rvec)
    print 'rotation', rotation, 'Rz angle = ', rotation[2] 
    angZ = rotation[2] 
    Rz = np.array([[math.cos(angZ), -math.sin(angZ), 0],[math.sin(angZ), math.cos(angZ), 0], [0, 0, 1]])
    projM = Rz.transpose()
    return projM

def calProjMatrix(cam, rvec, T):
    RzT = calRz(rvec)
    
    Tt = T.reshape((3,1))
    tvec = np.dot(-rvec,Tt)
    
    return RzT, rvec, tvec

def getCamMatrixs():
    cam = 0
    #calibration setup

    R11=0.837246
    R12=0.540749
    R13=0.0813053

    R21=0.285137
    R22=-0.558591
    R23=0.778892

    R31=0.466602
    R32=-0.628941
    R33=-0.621865

    T1=8.00144
    T2=6.79076
    T3=2.87561

    f=701.195
    mu=1
    mv=0.997588

    u0=485.301
    v0=432.17

    k1=-0.307813
    k2=0.112571
    k3=0
    p1=0
    p2=0
    
    rvec = np.array([[R11, R12, R13],[R21, R22, R23],[R31, R32, R33]])
    T = np.array([T1, T2, T3])
    camI0 = np.array([[-f*mu, 0, u0],[0,-f*mv, v0],[0,0,1]])
    distC0 = np.array([k1, k2, p1, p2, k3])
    RzT0, rvec0, tvec0 = calProjMatrix(cam, rvec, T)
    
    cam = 1
    #calibration setup

    R11=0.022185
    R12=-0.999565
    R13=-0.0194491

    R21=-0.615689
    R22=-0.0289869
    R23=0.787456

    R31=-0.787677
    R32=-0.00549512
    R33=-0.616064

    T1=15.0749
    T2=4.23487
    T3=2.89125

    f=695.313
    mu=1
    mv=0.997147

    u0=543.21
    v0=375.81

    k1=-0.327372
    k2=0.13491
    k3=0
    p1=0
    p2=0
    
    rvec = np.array([[R11, R12, R13],[R21, R22, R23],[R31, R32, R33]])
    T = np.array([T1, T2, T3])
    camI1 = np.array([[-f*mu, 0, u0],[0,-f*mv, v0],[0,0,1]])
    distC1 = np.array([k1, k2, p1, p2, k3])
    RzT1, rvec1, tvec1 = calProjMatrix(cam, rvec, T)
    
    cam = 2
    #calibration setup

    R11=-0.999042
    R12=-0.0363701
    R13=-0.0243473

    R21=-0.0412737
    R22=0.597806
    R23=0.800578

    R31=-0.0145621
    R32=0.800815
    R33=-0.598734

    T1=11.849
    T2=0.283563
    T3=2.79728

    f=669.93
    mu=1
    mv=0.997553

    u0=509.605
    v0=358.812

    k1=-0.308965
    k2=0.112368
    k3=0
    p1=0
    p2=0
    
    rvec = np.array([[R11, R12, R13],[R21, R22, R23],[R31, R32, R33]])
    T = np.array([T1, T2, T3])
    camI2 = np.array([[-f*mu, 0, u0],[0,-f*mv, v0],[0,0,1]])
    distC2 = np.array([k1, k2, p1, p2, k3])
    RzT2, rvec2, tvec2 = calProjMatrix(cam, rvec, T)
    
    cam = 3
    #calibration setup

    R11=-0.638223
    R12=0.767257
    R13=0.0631593

    R21=0.510238
    R22=0.360137
    R23=0.780999

    R31=0.57648
    R32=0.530677
    R33=-0.621331

    T1=5.66734
    T2=0.244033
    T3=2.85218

    f=674.319
    mu=1
    mv=0.995272

    u0=461.743
    v0=398.09

    k1=-0.316926
    k2=0.115751
    k3=0
    p1=0
    p2=0
    
    rvec = np.array([[R11, R12, R13],[R21, R22, R23],[R31, R32, R33]])
    T = np.array([T1, T2, T3])
    camI3 = np.array([[-f*mu, 0, u0],[0,-f*mv, v0],[0,0,1]])
    distC3 = np.array([k1, k2, p1, p2, k3])
    RzT3, rvec3, tvec3 = calProjMatrix(cam, rvec, T)
    
    RzTs = [RzT0, RzT1, RzT2, RzT3]
    rvecs = [rvec0, rvec1, rvec2, rvec3]
    tvecs = [tvec0, tvec1, tvec2, tvec3]
    camIs = [camI0, camI1, camI2, camI3]
    distCs = [distC0, distC1, distC2, distC3]
    return RzTs, rvecs, tvecs, camIs, distCs