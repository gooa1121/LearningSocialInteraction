import math
import numpy as np
import matplotlib
matplotlib.use("nbagg")
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import copy

def angle_rads(axis, vecs):
    vecs = np.array(vecs)
    angles = []
    for vec in vecs:
        angle = np.math.atan2( np.linalg.det([axis,vec]), np.dot(axis,vec) )
        angles.append(angle)
    return np.array(angles)

def pointsOnUnitWithCenter(pts, ori, scale = 1.): #input N*2 array and output N*2
    end_pts = np.zeros(pts.shape)
    # print end_pts.shape
    ori = ori.reshape((-1,1))
    end_pts[:,0] = pts[:,0] + np.cos(ori)[:,0]*scale
    end_pts[:,1] = pts[:,1] + np.sin(ori)[:,0]*scale     
    return end_pts

def pointsOnUnit(ori): #output N*3
    end_pts = np.zeros((len(ori),3))
    ori = ori.reshape((-1,1))
    end_pts[:,0] = np.cos(ori)[:,0]*1.
    end_pts[:,1] = np.sin(ori)[:,0]*1.     
    return end_pts

def projBack2World(pts, P):#pts shape should be 3*N, output = 3*N
    P = P.transpose()
    pts_w = np.dot(P, pts)
    return pts_w

def world2camRz(pts, Rz):#pts shape should be 3*N, output = 3*N
    ptsCam = np.dot(Rz, pts)
    return ptsCam
    
def world2Img(rvec, tvec, camI, distC, pts):
    outPoints = cv2.projectPoints(pts, rvec, tvec, camI, distC)
    pts_img = outPoints[0]
    pts_img = pts_img.reshape(pts_img.shape[0],2)
    return pts_img

def cam2Img(pts, Rz, rvec, tvec, camI, distC):
    pts_world = projBack2World(pts, Rz)
    pts_img = world2Img(rvec, tvec, camI, distC, pts_world.transpose() )
    return pts_img

def worldAng2ImgAng(angles, rvec, tvec, camI, distC):
    pts_world = np.tile( np.array([10.5, 3.5, 0]), (len(angles), 1) )#np.zeros((len(angles), 3))
    vec_world = pointsOnUnitWithCenter(pts_world[:,0:2], angles, scale = 2.)
    vec_world = np.concatenate((vec_world, np.zeros((vec_world.shape[0],1))), axis=1)
    endPts_Img = world2Img(rvec, tvec, camI, distC, vec_world)
    pts_Img = world2Img(rvec, tvec, camI, distC, pts_world)
    vec_Img = endPts_Img - pts_Img
    angles = angle_rads(np.array([1,0]), vec_Img)
    return angles

