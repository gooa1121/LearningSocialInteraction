import random
import numpy as np

import munkres
from munkres import Munkres, print_matrix
import codecs
import json
import copy 
import Config

#To remove!

import matplotlib
matplotlib.use("nbagg")
import skimage
import skimage.filters
import skimage.morphology
import math
import numpy.linalg as la
import numpy as np
import matplotlib.pyplot as plt
import cv2


class POM_evaluator(object):
    
    def __init__(self,room,GT_labels_path_json = '../../dataset/ETHZ/json_label/%08d.json',HW_GT_grid = (1240,480),shiftX_GT = 130,radius_match = 5, q_thresh = 0.5):
        self.room = room

        self.GT_labels_path_json = GT_labels_path_json
        self.H_GT_grid,self.W_GT_grid = HW_GT_grid[0],HW_GT_grid[1]
        self.shiftX = shiftX_GT
        self.radius_match = radius_match # This is with respect to the room resolution
        self.q_thresh = q_thresh


    #For ETH dataset
    def get_GT_coordinates_fromjson(self,fid):
        #ShiftX is a momnetaneous hack to match detection and labelling after modification
        '''
        Input: frame ID as in original dataset
        Output: coordinates with respect to the resolution of detection (H_grid,W_grid)
        '''
        room = self.room
        
        obj_text =codecs.open(self.GT_labels_path_json%fid, 'r', encoding='utf-8').read()
        print self.GT_labels_path_json%fid
        b_new = json.loads(obj_text)
        c_new = np.int32(b_new[1:])    

        n_cams = 7
        detections =[[] for cam in range(room.n_cams)]
        GT_coordinates = []
        # for rec in b_new:
        #     rec_ID = rec['positionID']
        #     y = np.round(-0.0 + (rec_ID/self.W_GT_grid))
        #     x = np.round(-0.0 + (rec_ID%self.W_GT_grid))
        #     if y > 0 and x >0 :
        #         GT_coordinates.append((y,x))
                #print 'recid, y, x:', rec_ID, y, x
        #print GT_coordinates
        print 'self.W_GT_grid = ', self.W_GT_grid,', self.H_GT_grid = ', self.H_GT_grid
        print 'room W = ', room.W_grid,', room H = ',room.H_grid
        
        for det_id in range(c_new.shape[0]):
            rec_ID = c_new[det_id,0]
            #print rec_ID
            if (rec_ID/self.W_GT_grid - self.shiftX)*(1.0*room.H_grid)/(1.0*self.H_GT_grid)  > -1 :
                GT_coordinates.append(((rec_ID/self.W_GT_grid- self.shiftX)*(1.0*room.H_grid)/(1.0*self.H_GT_grid),(rec_ID%self.W_GT_grid)*(1.0*room.W_grid)/(1.0*self.W_GT_grid)))
        # print GT_coordinates
        return np.asarray(GT_coordinates)
    
    #For Terrace Dataset
    def get_GT_coordinates_terrace(self,fid):
        #ShiftX is a momnetaneous hack to match detection and labelling after modification
        '''
        Input: frame ID as in original dataset
        Output: coordinates with respect to the resolution of detection (H_grid,W_grid)
        '''
        room = self.room
        
        all_detections = np.loadtxt('./terrace/gt_terrace1.txt')
        detections = all_detections[fid]
        GT_coordinates = []

        for det_id in range(detections.shape[0]):
                rec_ID = detections[det_id]
                #print rec_ID
                if rec_ID > -1:
                    x = rec_ID%room.H_grid
                    y = room.W_grid - rec_ID/room.H_grid

                    #HACK GT
                    # !!!! KEEP PARAMETERS!!!
                    #x = np.round(0.82*x + 2)
                    #y = np.round(0.77*y + 5)
                    x = np.round(0.89*x + 2)
                    y = np.round(0.82*y + 5)
                    
                    GT_coordinates.append((x,y))

        return np.asarray(GT_coordinates)

    
    #For Terrace Dataset
    def get_GT_coordinates_SALSA(self,fid,showWarning = True):
        #ShiftX is a momnetaneous hack to match detection and labelling after modification
        '''
        Input: frame ID as in original dataset (15 fps)
        Output: coordinates with respect to the resolution of detection (H_grid,W_grid)
        '''
        room = self.room
        
        all_detections = np.loadtxt('../../../dataset/Salsa/groundtruth.txt')
        # print 'all_detection shape', all_detections.shape
        if (fid - 3 )%45 != 0 and showWarning:
            print "for accurate ground truth , use fid = 3 + i*45"
        
        gt_line = (fid - 3) / 45
        
        detections = all_detections[:,gt_line]
        GT_coordinates = []

        for det_id in range(detections.shape[0]):
                
                rec_ID = detections[det_id]
                #print rec_ID
                if rec_ID > -1:
                    y = rec_ID%room.H_grid
                    x = rec_ID/room.H_grid
                    
                    GT_coordinates.append((x,y))

        return np.asarray(GT_coordinates)

    
    def hungarian_matching(self,GT_coordinates,det_coordinates,verbose = False):
        n_dets = det_coordinates.shape[0]
        n_gts = GT_coordinates.shape[0]
        print 'n_dets = %d, n_gts = %d' %(n_dets, n_gts)

        n_max = max(n_dets,n_gts)

        matrix = np.zeros((n_max,n_max)) + 1

        for i_d in range(n_dets):
            for i_gt in range(n_gts):
                if ((det_coordinates[i_d,0] - GT_coordinates[i_gt,0])**2 + (det_coordinates[i_d,1] - GT_coordinates[i_gt,1])**2) <= self.radius_match**2:
                    matrix[i_d,i_gt] = 0

        m = Munkres()
        indexes = m.compute(copy.copy(matrix))

        total = 0
        TP = [] #True positive
        FP = [] #False positive
        FN = [] #False negative
        for row, column in indexes:
            value = matrix[row][column]
            total += value
            if verbose:
                print '(%d, %d) -> %d' % (row, column, value)
            if value == 0:
                TP.append((int(det_coordinates[row,0]),int(det_coordinates[row,1])))
            if value >0:
                if row < n_dets:
                    FP.append((int(det_coordinates[row,0]),int(det_coordinates[row,1])))
                if column < n_gts :
                    FN.append((int(GT_coordinates[column,0]),int(GT_coordinates[column,1])))
        if verbose:
            print 'total cost: %d' % total

        return total,np.asarray(TP),np.asarray(FP),np.asarray(FN)


    def get_loss_arrays(self,TP,FP,FN):
        H_grid, W_grid = self.room.H_grid,self.room.W_grid
        LossMask = np.zeros(H_grid*W_grid)
        LossMask[W_grid*TP[:,0] + TP[:,1] ] = 1
        LossMask[W_grid*FP[:,0] + FP[:,1] ] = 1
        LossMask[W_grid*FN[:,0] + FN[:,1] ] = 1

        LossLabels = np.zeros(H_grid*W_grid)
        LossLabels[W_grid*TP[:,0] + TP[:,1] ] = 1
        LossLabels[W_grid*FP[:,0] + FP[:,1] ] = 0
        LossLabels[W_grid*FN[:,0] + FN[:,1] ] = 1

        return LossLabels,LossMask

    def get_Hungarian_score(self,Q_out,fid, show = False):
        GT_coordinates = self.get_GT_coordinates_fromjson(fid)
        det_coordinates = self.room.get_coordinates_from_Q(Q_out[-1],q_thresh = self.q_thresh)
        total,TP,FP,FN = self.hungarian_matching(GT_coordinates,det_coordinates,verbose = False)
        #print det_coordinates
        
        if show:
            MAP_out = np.zeros((self.room.H_grid,self.room.W_grid))
            # for (i,j) in GT_coordinates:
            #     MAP_out[int(i),int(j)] = 1
            # print 'GT'
            # plt.imshow(MAP_out)
            # plt.colorbar()
            # plt.show()
            
            # for (i,j) in det_coordinates:
            #     MAP_out[int(i),int(j)] += 2
                
            for (i,j) in GT_coordinates:
                #MAP_out[int(i),int(j)] = 1
                i = int(i)
                j = int(j)
                for yi in range( max(0,i-2), min(i+3, MAP_out.shape[0]) ):
                    for xj in range( max(0,j-2), min(j+3, MAP_out.shape[1]) ):
                        MAP_out[int(yi),int(xj)] = 1 
            print 'GT'
            plt.imshow(MAP_out)
            plt.colorbar()
            plt.show()
            
            #MAP_out = np.zeros((self.room.H_grid,self.room.W_grid))
            for (i,j) in det_coordinates:
                #MAP_out[int(i),int(j)] = 1
                i = int(i)
                j = int(j)
                for yi in range( max(0,i-2), min(i+3, MAP_out.shape[0]) ):
                    for xj in range( max(0,j-2), min(j+3, MAP_out.shape[1]) ):
                        MAP_out[int(yi),int(xj)] += 2
                        
            print 'det'
            plt.imshow(MAP_out)
            plt.colorbar()
            plt.show()


        return total,TP.shape[0],FP.shape[0],FN.shape[0], GT_coordinates, det_coordinates
    
    def get_Hungarian_score_terrace(self,Q_out,fid,show = False):
        GT_coordinates = self.get_GT_coordinates_terrace(fid)
        det_coordinates = self.room.get_coordinates_from_Q(Q_out[-1],q_thresh = self.q_thresh)
        total,TP,FP,FN = self.hungarian_matching(GT_coordinates,det_coordinates,verbose = False)
        
        if show:
            MAP_out = np.zeros((self.room.H_grid,self.room.W_grid))
            for (i,j) in GT_coordinates:
                MAP_out[int(i),int(j)] = 1
            for (i,j) in det_coordinates:
                MAP_out[int(i),int(j)] += 2

            plt.imshow(MAP_out)
            plt.colorbar()
            plt.show()

        return total,TP.shape[0],FP.shape[0],FN.shape[0]
    
    def get_Hungarian_score_SALSA(self,Q_out,fid,show = False,showWarning = True):
        GT_coordinates = self.get_GT_coordinates_SALSA(fid,showWarning)
        det_coordinates = self.room.get_coordinates_from_Q(Q_out[-1],q_thresh = self.q_thresh) #(y,x)
        total,TP,FP,FN = self.hungarian_matching(GT_coordinates,det_coordinates,verbose = False)
        
        if show:
            MAP_out = np.zeros((self.room.H_grid,self.room.W_grid))
            for (i,j) in GT_coordinates:
                MAP_out[int(i),int(j)] = 1
            # cv2.imwrite('groundTruth.png', MAP_out*255)
            for (i,j) in det_coordinates:
                MAP_out[int(i),int(j)] += 2

            plt.imshow(MAP_out)
            plt.colorbar()
            plt.show()
            # cv2.imwrite('ground.png', MAP_out*100)

        return total,TP.shape[0],FP.shape[0],FN.shape[0],GT_coordinates, det_coordinates


    
        
    
    #Read GT
    #Carefull in coordinates, j is before i


#     def get_GT_coordinates_fromtxt(self,fid):
#         '''
#         Input: frame ID as in original dataset
#         Output: coordinates with respect to the resolution of detection (H_grid,W_grid)
#         '''

#         f = open(GT_labels_path%img_index_list[fid], 'r')
#         n_cams = 7
#         detections =[[] for cam in range(n_cams)]
#         GT_coordinates = []
#         for lid,line in enumerate(f):
#             if lid >1:
#                 line_tab = np.asarray(np.fromstring(line, dtype=int, sep=','))
#                 rec_ID = line_tab[0]
#                 #print rec_ID
#                 GT_coordinates.append(((rec_ID/W_GT_grid)*(1.0*H_grid)/(1.0*H_GT_grid),(rec_ID%W_GT_grid)*(1.0*W_grid)/(1.0*W_GT_grid)))

#         return np.asarray(GT_coordinates)
