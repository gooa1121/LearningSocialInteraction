import math
import numpy as np
import matplotlib
matplotlib.use("nbagg")
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image

from pom_room import POM_room
import Config
import copy

def createMap(pts_list): # a list of several lists of points
    pts_np = np.array(pts_list) #L*N*2
    
    #find smallest x y in the list
    minPtsInList = np.amin(pts_np ,axis = 1) # L * 2 (2 = x,y)
    minxy = np.amin(minPtsInList, axis = 0)
    minxy = minxy.reshape(1,-1)
    # print 'shift x y amount = ', shiftx, shifty
    
    margin = 5
    shiftxy =  np.repeat(-minxy+margin, pts_np.shape[1], axis=0)
    pts_np[:] += shiftxy
    
    #find largest x y after shifting
    maxPtsInList = np.amax(pts_np ,axis = 1)
    max_xy = np.amax(maxPtsInList, axis = 0)
    [sizex, sizey] = max_xy

    ground = np.zeros(( int(sizey)+margin, int(sizex)+margin ))
    # print 'create a map with size %d x %d'%(int(sizey)+r, int(sizex)+r )
    return ground, pts_np

def drawLinesOnGround(pts, end_pts_List, scale):
    #scale for better visualize
    pts = np.array(pts[:,0:2])*scale
    ptsList = []
    ptsList.append(pts)
    for end_pts in end_pts_List:
        end_pts = np.array(end_pts[:,0:2])*scale
        ptsList.append(end_pts)
    ground, all_pts_shift = createMap(ptsList)
    
    pos_pts_shift = all_pts_shift[0]
    end_pts_shift = all_pts_shift[1:]
    
    for end_pts in end_pts_shift:
        for idx, (p, ep) in enumerate(zip(pos_pts_shift, end_pts)):
            p = p.astype(int)
            ep = ep.astype(int)
            cv2.circle(ground, (p[0], p[1]), 1, (255,0,0), -1)
            cv2.arrowedLine(ground, (p[0], p[1]), (ep[0], ep[1]), (180, 0, 0), 1)
            cv2.putText(ground,str(idx+1),(p[0], p[1]), cv2.FONT_HERSHEY_PLAIN, 0.5,(100,0,0), thickness = 1)
    return ground, pos_pts_shift, end_pts_shift

def OverlayOnGround(ground, pts, end_pts):
    for idx, (p, ep) in enumerate(zip(pts, end_pts)):
        p = p.astype(int)
        ep = ep.astype(int)
        cv2.circle(ground, (p[0], p[1]), 1, (2,0,0), -1)
        cv2.putText(ground,str(idx+1),(p[0], p[1]), cv2.FONT_HERSHEY_PLAIN, 0.5,(2,0,0), thickness = 1)
        cv2.line(ground, (p[0], p[1]), (ep[0], ep[1]), (1, 0, 0), 1)
    return ground

    
def drawLinesInImg(pts, pts2, in_img, lineColor, text=True):
    img = copy.copy(in_img)
    for idx, (pt, pt2) in enumerate(zip(pts, pts2)):
        x = int(pt[0])
        y = int(pt[1])
        cv2.circle(img, (x, y), 5, (255,0,0), -2)
        if text:
            cv2.putText(img,str(idx+1),(x, y), cv2.FONT_HERSHEY_PLAIN, 3,(255,0,0), thickness = 3)

        x2_b = int(pt2[0])
        y2_b = int(pt2[1])
        cv2.arrowedLine(img, (x,y), (x2_b, y2_b), lineColor, 2)
    return img

def drawMembershipImg(rgbImg, membersGT, x_img, y_img):#(rgbImg, membersGT, mID, x_img, y_img):
    # rgbFile = Config.rgb_name_list[cam]%(Config.img_index_list[fid])
    # rgbImg = Image.open(rgbFile)
    # rgbImg = np.array(rgbImg)
    Hrgb,Wrgb = rgbImg.shape[0:2]
    rgbImg = copy.copy(rgbImg) 
    
    # x_img = x_img.reshape((len(x_img),-1))
    # y_img = y_img.reshape((len(y_img),-1))
    ##loop for each group
    for idx, members in enumerate(membersGT):
        members = np.array(members)
        membersIdx = members - 1
        print members
        print x_img.shape
        members_x = x_img[membersIdx]
        members_y = y_img[membersIdx]
        centerx = np.mean(members_x)
        centery = np.mean(members_y)
        centerx = int(centerx)
        centery = int(centery)
        #assign group color
        r = 80 * ((idx+3)/3) *( (idx)%3==0 ) #idx==0, 3
        g = 80 * ((idx+3)/3) *( (idx)%3==1  ) #idx==1
        b = 80 * ((idx+3)/3) *( (idx)%3==2 ) #idx==2
        #visualize group center
        cv2.circle(rgbImg, (centerx, centery), 5, (r,g,b), -2)
        sign = 'Group %d'%(idx+1)
        if len(members)<2:
            sign = '  alone'
        cv2.putText(rgbImg, sign, (centerx, centery), cv2.FONT_HERSHEY_PLAIN, 2,(r,g,b),2,cv2.LINE_AA)
        #draw group and connection line
        for id, (x, y, m) in enumerate(zip(members_x, members_y, members)):
            x = int(x)
            y = int(y)
            cv2.circle(rgbImg, (x, y), 5, (r,g,b), -2)
            cv2.line(rgbImg, (x, y), (centerx, centery), (r, g, b), 2)
            #cv2.putText(rgbImg, str(m+1), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5,(r,g,b),2,cv2.LINE_AA)
            # cv2.putText(rgbImg, str(int(mID[m][0])), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5,(r,g,b),2,cv2.LINE_AA)
            cv2.putText(rgbImg, str(int(m)), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5,(r,g,b),2,cv2.LINE_AA)
    
    return rgbImg

def drawBoxMembershipImg(membersGT, mID, bboxes, x_img, y_img, fid, cam):
    rgbFile = Config.rgb_name_list[cam]%(Config.img_index_list[fid])
    rgbImg = Image.open(rgbFile)
    rgbImg = np.array(rgbImg)
    Hrgb,Wrgb = rgbImg.shape[0:2]
    rgbImg = copy.copy(rgbImg) 
    ##loop for each group
    for idx, (members, memberIDs) in enumerate(zip(membersGT, mID)):
        #assign group color
        r = 80 * ((idx+3)/3) *( (idx)%3==0 ) #idx==0, 3
        g = 80 * ((idx+3)/3) *( (idx)%3==1  ) #idx==1
        b = 80 * ((idx+3)/3) *( (idx)%3==2 ) #idx==2
        
        members = np.array(members)
        
        #visualize group center
        members -= 1
        members_x = x_img[members]
        members_y = y_img[members]
        centerx = int( np.mean(members_x) )
        centery = int( np.mean(members_y) )
        cv2.circle(rgbImg, (centerx, centery), 5, (r,g,b), -2)
        sign = 'Group %d'%(idx+1)
        if len(members)<2:
            sign = '  alone'
        cv2.putText(rgbImg, sign, (centerx, centery), cv2.FONT_HERSHEY_PLAIN, 3,(r,g,b),2,cv2.LINE_AA)
        
        #draw group and connection line
        print 'members',members, 'Ids', memberIDs
        for id, (x, y, m, ID) in enumerate(zip(members_x, members_y, members, memberIDs)):
            x = int(x)
            y = int(y)
            cv2.circle(rgbImg, (x, y), 5, (r,g,b), -2)
            cv2.line(rgbImg, (x, y), (centerx, centery), (r, g, b), 2)
            #cv2.putText(rgbImg, str(m+1), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5,(r,g,b),2,cv2.LINE_AA)
            cv2.putText(rgbImg, str(int(ID)), (x, y), cv2.FONT_HERSHEY_PLAIN, 3,(255,0,0),2,cv2.LINE_AA)
            bbox = np.asarray(bboxes[m]).astype(np.int)
            cv2.rectangle(rgbImg,(Config.CNN_factor*bbox[1],Config.CNN_factor*bbox[2]),
                          (Config.CNN_factor*bbox[3],Config.CNN_factor*bbox[4]),(r,g,b),3)
    
    # plt.figure(figsize=(10,10))
    # plt.imshow(rgbImg)
    # plt.show()
    return rgbImg

def pairImgs(img1, img2, saveImg = False):
    fig = plt.figure(figsize=(20,10))
    axis1 = fig.add_subplot(121)
    plt.imshow(img1)
    axis2 = fig.add_subplot(122)
    plt.imshow(img2)
    plt.show()
    
    if saveImg:
        plt.imsave('ori_GT_fid%d_cam%d.png'%(fid,cam), groundGT)
        plt.imsave('ori_estimate_GT_fid%d_cam%d.png'%(fid,cam), groundEST)
