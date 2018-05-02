import pickle
import numpy as np
import os
from PIL import Image
import json
from scipy import ndimage
import math
import random
import cv2
import matplotlib.pyplot as plt
import copy
from munkres import Munkres

from UnariesNet_orien import unariesNet
import MyConfig_orien as MyConfig
import Config_orien as Config
import getCamM
import geoFuncs
import drawLib

def AddPatchToImg(Img, centerInImg, patch):
    c = centerInImg.astype('int')
    kx0 = ky0 = 0
    ky1,kx1 = patch.shape

    x0 = max(0,c[0]-(kx1/2))
    y0 = max(0, c[1]-(ky1/2))
    x1 = min(Img.shape[1],c[0]+(kx1/2)+1)
    y1 = min(Img.shape[0], c[1]+(ky1/2)+1)
    
    #deal with border case
    if x0 == 0 and x0 != c[0]-(kx1/2):
        kx0 = kx1/2 - c[0]
    if x1 == Img.shape[1] and x1!=c[0]+(kx1/2)+1:
        kx1 = kx1/2 + Img.shape[1]-c[0]
        
    
    if y0 == 0 and y0 != c[1]-(ky1/2):
        ky0 = ky1/2 - c[1]
    
    if y1 == Img.shape[0] and y1!=c[1]+(ky1/2)+1:
        ky1 = ky1/2 + Img.shape[0]-c[1]

    Img[y0:y1, x0:x1] += patch[ky0:ky1, kx0:kx1]
    return Img

def addOvalGaussian(voteMap, CountMap, sigma, c, theta, threshold = 0.4, kernel_size = 19):
    mu = [0,0]
    ax = np.linspace(-7, 7, kernel_size) 
    ay = np.linspace(-7, 7, kernel_size)
    x, y = np.meshgrid(ax, ay)

    A = 1.0
    theta = -theta
    a = (math.cos(theta)*math.cos(theta))/(2*sigma[0]*sigma[0]) + (math.sin(theta)*math.sin(theta))/(2*sigma[1]*sigma[1])
    b = -math.sin(2*theta)/(4*sigma[0]*sigma[0]) + math.sin(2*theta)/(4*sigma[1]*sigma[1])
    c = (math.sin(theta)*math.sin(theta))/(2*sigma[0]*sigma[0]) + (math.cos(theta)*math.cos(theta))/(2*sigma[1]*sigma[1])
    kernel = A*np.exp( - (a*np.power(x-mu[0],2) + 2*b*(x-mu[0])*(y-mu[1]) + c*np.power((y-mu[1]),2) ))
    #thresholding for countMap
    b_kernel = kernel>threshold
    
#     plt.imshow(kernel)
#     plt.colorbar()
#     plt.show()
#     print center
    voteMap = AddPatchToImg(voteMap, c, kernel)
    CountMap= AddPatchToImg(CountMap, c, b_kernel)
    return voteMap, CountMap

def addPolarGaussian(voteMap, CountMap, sigma, c, theta, r0 = 1, threshold = 0.4, kernel_size = 19):
    x = np.linspace(-4, 4, kernel_size) 
    y = np.linspace(-4, 4, kernel_size) 
    X, Y = np.meshgrid(x, y)

    Xp = X*np.cos(theta) + Y*np.sin(theta)
    Yp = -X*np.sin(theta) + Y*np.cos(theta)
    #r and theta
    amp=1.
    r0 = 1
    kernel = amp * np.exp(- ( ((X**2+Y**2)-2*((X**2+Y**2)**0.5)*r0+r0**2 )/((sigma[0]**2)*2) + ((np.arctan2(Yp,Xp))**2)/((sigma[1]**2)*2) ))
    b_kernel = kernel>threshold

    voteMap = AddPatchToImg(voteMap, c, kernel)
    CountMap= AddPatchToImg(CountMap, c, b_kernel)

    return voteMap, CountMap

def buildMaps(ground, pts, ang_b, sigmaR, sigmaA, kSize, r_mu=1.0, Vthreshold=0.5):
    voteMaps = np.zeros((len(pts), ground.shape[0], ground.shape[1]))
    voteCountMaps = np.zeros( (len(pts), ground.shape[0], ground.shape[1]) )
    for idx, (p, ori) in enumerate(zip(pts, ang_b)):
        sigma = [sigmaR, sigmaA]
        l = int(kSize*2)+1
        voteMaps[idx], voteCountMaps[idx] = addPolarGaussian(voteMaps[idx], voteCountMaps[idx], sigma, p, ori, r0 = r_mu, threshold = Vthreshold, kernel_size = l)
    return voteMaps, voteCountMaps

def Voting(voteMaps, voteCountMaps, show = False, way = 'region'):
    roundN = 0
    accumVoteMap = np.sum(voteMaps, axis = 0)
    accumCountMap = np.sum(voteCountMaps, axis = 0)
    if show:
        print 'accum map'
        drawLib.pairImgs(accumVoteMap, accumCountMap)
    
    memberShip = []
    max_region_scale = 1
    
    #rounds of voting result 
    while np.sum(accumCountMap)!=0:
        weightedAccum = accumVoteMap * accumCountMap
        if show:
            print 'weighted accum map'
            # plt.imsave('weightedAccum%d.png'%roundN, weightedAccum)
            fig = plt.figure(figsize=(10,10))
            plt.imshow(weightedAccum)
            plt.colorbar()
            plt.show()
        #find local max and its region
        maxV = np.amax(weightedAccum) 
        max_pos = np.unravel_index(weightedAccum.argmax(), weightedAccum.shape)
        votePeople = accumCountMap[max_pos[0], max_pos[1]]
#         print 'max value', maxV, ' in ', max_pos, ' with voting num', votePeople
        
        maxRegionMap = np.zeros(weightedAccum.shape)
        if way == 'center':
            maxRegionMap[ max_pos[0],  max_pos[1]] = 1
        else:   
            regionR = int(max_region_scale * votePeople)
            cv2.circle(maxRegionMap, (max_pos[1], max_pos[0]), regionR, (1,0,0), -1)
        if show:
            print 'region' 
            plt.imshow(maxRegionMap*6 + accumVoteMap)
            plt.show()
        #find who's voting to the region and remove them from voting rounds
        memberShip_round = []
        if way == 'center':
            thre = 1 
        else:
            thre = 1#float(np.sum(maxRegionMap))/(float(votePeople)*3.6)#need to set some value around or more than pi*r
            if votePeople == 1:
                thre = 1     
                # print 'thre = ', thre
        for idx, personalMap in enumerate(voteCountMaps):
            votingScore = maxRegionMap * personalMap

            if np.sum(votingScore) >= thre:
                memberShip_round.append(idx+1)
                #remove personal factor from the voting space
                voteCountMaps[idx] = np.zeros(accumCountMap.shape)
                voteMaps[idx] = np.zeros(accumVoteMap.shape)
        memberShip.append(memberShip_round)
        
        #prepare for next round
        accumVoteMap = np.sum(voteMaps, axis = 0)
        accumCountMap = np.sum(voteCountMaps, axis = 0)
        if show:
            print 'roundN', roundN, 'memberShip:', memberShip_round
            print 'remain for next round'
            drawLib.pairImgs(accumVoteMap, accumCountMap)

        roundN += 1
    return memberShip

def evaluate(GTList, ESTList, T = 2./3.):
    TP = 0
    FP = 0
    FN = 0
    
    matchedEST = []
    updateESTList = ESTList
    for group in GTList:
        G = float(len(group))
        GTset = set(group)
        match = False
        min_thres = G*T
        max_thres = G*(1-T)
        
        #match with each estimate group
        restESTList = []
        for ESTgroup in updateESTList:
            ESTset = set(ESTgroup)
            intersect = GTset.intersection(ESTset)
            subtract = ESTset - GTset
            if len(intersect) >= min_thres and len(subtract) <= max_thres:
                TP += 1
                match = True
                matchedEST.append(ESTgroup)
            else:
                restESTList.append(ESTgroup)
        if not match:
            FN += 1
        updateESTList = restESTList
    
    print 'rest', updateESTList
    print 'matched', matchedEST
    FP = len(updateESTList)
    return TP, FP, FN

def hungarian_matching(GT_coordinates, det_coordinates, verbose = False):
    n_dets = det_coordinates.shape[0]
    n_gts = GT_coordinates.shape[0]
    print 'n_dets = %d, n_gts = %d' %(n_dets, n_gts)

    n_max = max(n_dets,n_gts)
    matrix = np.zeros((n_max,n_max)) + 1
    
    radius_match = 5
    for i_d in range(n_dets):
        for i_gt in range(n_gts):
            if ((det_coordinates[i_d,0] - GT_coordinates[i_gt,0])**2 + (det_coordinates[i_d,1] - GT_coordinates[i_gt,1])**2) <= radius_match**2:
                matrix[i_d,i_gt] = 0

    m = Munkres()
    indexes = m.compute(copy.copy(matrix))
    #print indexes
    
    total = 0
    TP = [] #True positive
    FP = [] #False positive
    FN = [] #False negative
    detMatchedID = np.zeros((n_dets,1))
    for row, column in indexes:
        value = matrix[row][column]
        total += value
        if verbose:
            print '(%d, %d) -> %d' % (row, column, value)
        if value == 0:
            TP.append((int(det_coordinates[row,0]),int(det_coordinates[row,1])))
            detMatchedID[row] = column+1 #GT given by salsa annotated from 1
        if value >0:
            if row < n_dets:
                FP.append((int(det_coordinates[row,0]),int(det_coordinates[row,1])))
                detMatchedID[row] = 0
            if column < n_gts :
                FN.append((int(GT_coordinates[column,0]),int(GT_coordinates[column,1])))

    return detMatchedID
