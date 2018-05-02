import numpy as np
import matplotlib.pyplot as plt
import cv2
import re, os, glob, pickle, shutil,sys, random, copy
from shutil import *

sys.path.append('../roi_pooling/theano-roi-pooling/')
sys.path.append('./POM')

from theano import *
theano.__version__
import theano
from theano import tensor as T

config.allow_gc =False

import copy
from PIL import Image

from pom_room import POM_room
from pom_evaluator import POM_evaluator

import Config
import VGG.VGGNet as VGGNet
from roi_pooling import ROIPoolingOp
from net_functions import *

import MyConfig_orien as MyConfig
import math

class unariesNet:
    def __init__(self,load_pretrained = True, training = True):
        #Path save params
        self.path_save_params = MyConfig.unaries_params_path
        print 'param save at = ',  self.path_save_params 

        #logs
        self.train_logs_path = MyConfig.unaries_train_log
        self.test_logs_path = MyConfig.unaries_test_log
        
        #Oputput
        self.unaries_out_path = Config.unaries_path
        
        print "Preparing room"
        #Prepare room and evaluator
        #Create room
        self.room = POM_room(Config.parts_root_folder,with_templates= True)
        #Prepare evaluator which will let us load GT
        self.evaluator = POM_evaluator(self.room,GT_labels_path_json = '../NDF_peds/data/ETH/labels_json/%08d.json')
        
        
        print "Initializing Unaries Network"
        #DEFINE NETWORK
        
        '''
        Remark, when using ROIPooling, y axis first then x axis for ROI pooling
        '''
        p_h,p_w = 3,3 #"size of extracted features vector"

        epsilon = 1e-7
        X = T.ftensor4('X')
        Ybb= T.fvector('Ybb')# GT for positive or negative bbox
        Ybody= T.fvector('Ybody')
        Yhead= T.fvector('Yhead')
        
        batch_size = X.shape[0]
        p_drop = T.scalar('dropout',dtype = 'float32')
        t_rois = T.fmatrix()

        # Building net
        ## Convnet 

        mNet = VGGNet.VGG(X)

        c53_r = mNet.c53_r

        op = ROIPoolingOp(pooled_h=p_h, pooled_w=p_w, spatial_scale=1.0)


        roi_features = op(c53_r, t_rois)[0]#T.concatenate(op(c53, t_rois),axis = 0)

        #Initialize weights
        w0_u = init_weights((512*p_h*p_w,1024),name = 'w0_unaries')
        b0_u = init_weights((1024,),name = 'b0_unaries',scale = 0)
        w1_u = init_weights((1024,1024),name = 'w1_unaries')
        b1_u = init_weights((1024,),name = 'b1_unaries',scale = 0)
        w2_u = init_weights((1024,2),name = 'w2_unaries')
        b2_u = init_weights((2,),name = 'b2_unaries',scale = 0)
        #for orientation of body, head estimation
        w2_u_ori = init_weights((1024,2),name = 'w2_unaries_ori')
        b2_u_ori = init_weights((2,),name = 'b2_unaries_ori',scale = 0)


        paramsUnaries = [w0_u,b0_u,w1_u,b1_u,w2_u,b2_u,w2_u_ori,b2_u_ori]


        # #New network
        features_flat = roi_features.reshape((-1,512*p_h*p_w))
        x1 = T.clip(T.dot(features_flat,w0_u) + b0_u,0,100000)
        x1_drop = dropout(x1,p_drop)
        x2 = T.clip(T.dot(x1_drop,w1_u) + b1_u,0,100000)
        x2_drop = dropout(x2,p_drop)
        p_out = softmax(T.dot(x2_drop,w2_u) + b2_u)
        log_p_out = stab_logsoftmax(T.dot(x2_drop,w2_u) + b2_u)
        #Another FC layer for orientation of body, head estimation
        rad_out = T.clip(T.dot(x2_drop,w2_u_ori) + b2_u_ori,-math.pi,math.pi)
        
        ## Classification
        # loss = -(log_p_out[:,0]*Ybb + log_p_out[:,1]*(1-Ybb)).mean()
        loss_bbox = -(log_p_out[:,0]*Ybb + log_p_out[:,1]*(1-Ybb)).mean()
        ## norm2
        # loss_body = (Ybb*(Ybody-rad_out[:,0])*(Ybody-rad_out[:,0])).mean()
        # loss_head = (Ybb*(Yhead-rad_out[:,1])*(Yhead-rad_out[:,1])).mean()
        ## norm1 
        # loss_body = (Ybb*abs(Ybody-rad_out[:,0])).mean()
        # loss_head = (Ybb*abs(Yhead-rad_out[:,1])).mean()
        ## norm1 with precise weight
        # loss_body = (Ybb*abs(Ybody-rad_out[:,0])).sum()/Ybb.sum()
        # loss_head = (Ybb*abs(Yhead-rad_out[:,1])).sum()/Ybb.sum()
        unit = 1.0
        est_body_orienX = unit*np.cos(rad_out[:,0])# x on th unit circle
        est_body_orienY = unit*np.sin(rad_out[:,0])# y on th unit circle
        gt_body_orienX = unit*np.cos(Ybody)
        gt_body_orienY = unit*np.sin(Ybody)
        d_bodyX = est_body_orienX - gt_body_orienX
        d_bodyY = est_body_orienY - gt_body_orienY
        cost_body = np.sqrt( d_bodyX*d_bodyX + d_bodyY*d_bodyY)
        
        est_head_orienX = unit*np.cos(rad_out[:,1])# x on th unit circle
        est_head_orienY = unit*np.sin(rad_out[:,1])# y on th unit circle
        gt_head_orienX = unit*np.cos(Yhead)
        gt_head_orienY = unit*np.sin(Yhead)
        d_headX = est_head_orienX - gt_head_orienX
        d_headY = est_head_orienY - gt_head_orienY
        cost_head = np.sqrt(d_headX*d_headX + d_headY*d_headY)
        
        loss_body = (Ybb*cost_body).sum()/Ybb.sum()
        loss_head = (Ybb*cost_head).sum()/Ybb.sum()
        
        lambda1 = 0.5
        lambda2 = 0.5
        # print loss_bbox, loss_head, loss_body
        loss = loss_bbox + lambda1*loss_body + lambda2*loss_head


        # Updates for decision parameter
        ## For regression tree/Flat
        updates_loss = Adam(loss,paramsUnaries,lr=2e-4)
        updates_loss_VGG = Adam(loss,paramsUnaries+mNet.paramsVGG,lr=1e-6)

        self.train_func = theano.function(inputs=[X,t_rois,Ybb, Ybody, Yhead,In(p_drop, value=0.5)], 
                                     outputs=[T.exp(log_p_out),loss, rad_out, loss_bbox, loss_body, loss_head], updates=updates_loss_VGG, allow_input_downcast=True,on_unused_input='warn')
        
        self.test_func = theano.function(inputs=[X,t_rois,Ybb, Ybody, Yhead,In(p_drop, value=0.0)],
                                    outputs=[T.exp(log_p_out),loss, rad_out, loss_bbox, loss_body, loss_head], updates=[],
                                    allow_input_downcast=True,on_unused_input='warn')
        
        self.run_func = theano.function(inputs=[X,t_rois,In(p_drop, value=0.0)],
                                   outputs=[T.exp(log_p_out),rad_out], updates=[],
                                   allow_input_downcast=True,on_unused_input='warn')
        
        self.play_func = theano.function(inputs=[X,t_rois,In(p_drop, value=0.0)],
                                    outputs=roi_features, updates=[],
                                    allow_input_downcast=True,on_unused_input='warn')
        
        self.features_func = theano.function(inputs=[X,t_rois,In(p_drop, value=0.0)],
                                   outputs=x2, updates=[],
                                   allow_input_downcast=True,on_unused_input='warn')
       
        
        #Define self objects
        self.paramsUnaries = paramsUnaries
        self.mNet = mNet
        
        #Load pretrained params
        if load_pretrained:
            print "loading pretrained params for bbox detection"
            print MyConfig.unary_storedParam
            params_to_load = pickle.load(open(MyConfig.unary_storedParam))
             #append the params for orientation estimation
            if training:
                print 'append value'
                params_to_load.append(floatX(np.random.randn(*(1024,2)) * 0.01))
                params_to_load.append(floatX(np.random.randn(*(2,)) * 0.0))
            self.setParams(params_to_load)
            
            print MyConfig.refinedVGG_storedParam
            params_VGG= pickle.load(open(MyConfig.refinedVGG_storedParam))
            mNet.setParams(params_VGG)

            
        
        
    def getParams(self):
        params_values = []
        for p in range(len(self.paramsUnaries)):
            params_values.append(self.paramsUnaries[p].get_value())

        return params_values

    def setParams(self,params_values):
        for p in range(len(params_values)):
            self.paramsUnaries[p].set_value(params_values[p])
            
    
    def train(self, resume_epoch = 0,fine_tune = True):
        print 'train orien unary'
        test_fid = 1

        if resume_epoch ==0:
            f_logs = open(self.train_logs_path,  'w')
            f_logs.close()
            f_logs = open(self.test_logs_path,  'w')
            f_logs.close()

        else:
            params_to_load = pickle.load(open(self.path_save_params + 'params_Unaries_%d.pickle'%(resume_epoch-1)))
            self.setParams(params_to_load)
            if fine_tune:
                params_VGG= pickle.load(open(self.path_save_params + 'params_VGG_%d.pickle'%(resume_epoch-1)))
                self.mNet.setParams(params_VGG)


        #load the orientation ground truth
        # self.GT_bodys = np.load('./GT_orien/GT_body_proj.npy')
        # self.GT_heads = np.load('./GT_orien/GT_head_proj.npy')
        self.GT_bodys = np.load('./GT_orien/GT_body_camSpace.npy')
        self.GT_heads = np.load('./GT_orien/GT_head_camSpace.npy')
        for epoch in range(resume_epoch,80):
            costs = []
            for fid in range(0,len(Config.img_index_list)):
                for cam in Config.cameras_list:
                    print 'Epoch %d, FID %d, cam %d'%(epoch,fid,cam)
                    x,rois_np,labels, body_labels, head_labels = self.load_batch_train(fid,cam)
                    # print 'roi=', rois_np.shape
                    #visualize_batch(x,rois_np,labels)
                    p_out_train,loss,estimate_rad, l_bb, l_b, l_h = self.train_func(x,rois_np,labels, body_labels, head_labels)
                    print 'cost: bbox, body, head:', l_bb, l_b, l_h
                    costs.append(loss)
                    #x_out_test = test_func(rgb_theano,rois_np)
            

            #Save params
            if epoch%5 ==0:
                if not os.path.exists( MyConfig.unaries_params_path):
                    os.makedirs( MyConfig.unaries_params_path )
                params_to_save  = self.getParams()
                pickle.dump(params_to_save,open(self.path_save_params  +'params_Unaries_%d.pickle'%epoch,'wb'))
                if fine_tune:
                    params_VGG  = self.mNet.getParams()
                    pickle.dump(params_VGG,open(self.path_save_params  +"params_VGG_%d.pickle"%epoch,'wb'))


            av_cost = np.mean(costs)
            f_logs = open(self.train_logs_path,  'a')
            f_logs.write('%f'%(av_cost) + '\n')
            f_logs.close()
            
                        #Test loss
            if test_fid > 0:
                test_costs = []
                fid = test_fid
                for cam in Config.cameras_list:
                        print 'Test Epoch %d, FID %d, cam %d'%(epoch,fid,cam)
                        x,rois_np,labels, body_labels, head_labels = self.load_batch_train(fid,cam)
                        print 'roi=', rois_np.shape
                        print 'labels=', labels.shape
                        p_out_test,test_loss, estimate_rad, l_bb, l_b, l_h = self.test_func(x,rois_np,labels,body_labels, head_labels)
                        print 'cost: bbox, body, head', l_bb, l_b, l_h
                        self.visualize_positives(x,rois_np,p_out_test, body_labels, head_labels, estimate_rad, i=fid, cam=cam)
                        test_costs.append(test_loss)


                av_test_cost = np.mean(test_costs)
                f_logs = open(self.test_logs_path,  'a')
                f_logs.write('%f'%(av_test_cost) + '\n')
                f_logs.close()
        # return rois_np,labels, body_labels, head_labels, select
        params_to_save  = self.getParams()
        pickle.dump(params_to_save,open(self.path_save_params  +'params_Unaries_%d.pickle'%epoch,'wb'))
        if fine_tune:
            params_VGG  = self.mNet.getParams()
            pickle.dump(params_VGG,open(self.path_save_params  +"params_VGG_%d.pickle"%epoch,'wb'))
                
    #FUNCTIONS TO LOAD DATA
    
    def get_rois(self,fid,cam):
        n_parts = Config.n_parts
        thresh =0.40
        #####
        #Loading the image preprocessed with segmentor
        templates_array = self.room.templates_array
        image = self.room.load_images_stacked(fid, verbose = False)

        indices = templates_array.shape[1]
        indices_reduced,scores = self.room.get_indices_above(image,threshold= thresh)
        templates_array_reduced = templates_array[:,indices_reduced,:]
        #####
        #Now we have preselected bboxes
        # print 'with enough fg ', templates_array_reduced.shape
        templates = templates_array_reduced[n_parts -1 + n_parts*cam]

        crit_no_null = (templates[:,2]-templates[:,0])*(templates[:,3]-templates[:,1]) > 400 #We don't want empty boxes
        templates_no_null = templates[crit_no_null]
        indices_no_null = indices_reduced[crit_no_null]
        
        if len(indices_no_null) == 0:
            crit_no_null = (templates[:,2]-templates[:,0])*(templates[:,3]-templates[:,1]) >= 20
            templates_no_null = templates[crit_no_null]
            indices_no_null = indices_reduced[crit_no_null]
            print '=====smaller threshold=====', templates_no_null.shape
        #if len(indices_no_null) == 0:
        #    templates_no_null = templates[0:2]
        #    indices_no_null = [0,1]
        #    print 'created', templates_no_null.shape
        #    print templates_no_null
        # rois fill
        rois_np = np.zeros((templates_no_null.shape[0],5)).astype(np.single)

        rois_np[:,1] = templates_no_null[:,1]
        rois_np[:,2] = templates_no_null[:,0]
        rois_np[:,3] = templates_no_null[:,3]
        rois_np[:,4] = templates_no_null[:,2]
        # print 'unique roi1', np.unique(rois_np[:,1])
        # print 'unique roi2', np.unique(rois_np[:,2])
        # print 'unique roi3', np.unique(rois_np[:,3])
        # print 'unique roi4', np.unique(rois_np[:,4])

        return rois_np,indices_no_null

    def get_rgb(self,fid,cam):
        #Load rgb image
        rgb = np.asarray(Image.open(Config.rgb_name_list[cam]%self.room.img_index_list[fid]))[:,:,0:3]
        H,W = np.shape(rgb)[0:2]
        rgb_theano = rgb.transpose((2,0,1))
        rgb_theano = rgb_theano.reshape((1,3,H,W))

        return rgb_theano

    def get_labels(self,fid,cam, indices_no_null,rad = 1 ):
        #rad = radius to validate a detection
        #Load ground_truth
        GT_coordinates = np.floor(self.evaluator.get_GT_coordinates_SALSA(fid)).astype(np.int)
        gt_line = (fid - 3) / 45
        print 'get label gt_line = ', gt_line
        body_GT_frame = self.GT_bodys[cam, gt_line, :]
        head_GT_frame = self.GT_heads[cam, gt_line, :]
        det_coordinates = self.room.get_coordinates_from_Q_reduced(indices_no_null*0 + 1.0,indices_no_null).astype(np.int)

        #Find positive examples
        MAP_OK = np.zeros((self.room.H_grid,self.room.W_grid))

        for X in GT_coordinates.tolist() :
            MAP_OK[X[0],X[1]] = 1
        
        #assign label of orientation    
        labels_body = []#np.zeros((det_coordinates.shape[0],1))-4
        labels_head = []#np.zeros((det_coordinates.shape[0],1))-4
        rad2 = 2
        for idx,X in enumerate(det_coordinates.tolist()):
            correspondGT = (GT_coordinates[:,0]>X[0]-rad2) * (GT_coordinates[:,0]<X[0]+rad2) * (GT_coordinates[:,1]>X[1]-rad2) * (GT_coordinates[:,1]<X[1]+rad2)
            GT_candidate_id = np.where(correspondGT)[0]
            
            if len(GT_candidate_id)>0:
                # print GT_candidate_id
                winner = GT_candidate_id[0]
                # if winner != 17:
                    # print np.where(correspondGT)
                if GT_candidate_id.shape[0] > 1:
                    # print 'pick one with shortest distance'
                    dist = (GT_coordinates[winner][0] - X[0])*(GT_coordinates[winner][0] - X[0]) + (GT_coordinates[winner][1] - X[1])*(GT_coordinates[winner][1] - X[1])
                    for GT_idx in GT_candidate_id[1:]:
                        # print GT_idx
                        newDist = (GT_coordinates[GT_idx][0] - X[0])*(GT_coordinates[GT_idx][0] - X[0]) + (GT_coordinates[GT_idx][1] - X[1])*(GT_coordinates[GT_idx][1] - X[1])
                        if dist > newDist:
                            winner = GT_idx
                            dist = newDist
                # print 'orien: ', body_GT_frame[winner]
                labels_body.append(body_GT_frame[winner])
                labels_head.append(head_GT_frame[winner])
            else:
                # print 'false detec'
                labels_body.append(-5)
                labels_head.append(-5)
                    

    #     plt.imshow(MAP_OK)
    #     plt.show()
        #Maybe overkill but will use integral image in order to computer afterward iintegral inside area for detections
        MAP_OK_integral = MAP_OK.cumsum(axis =0).cumsum(axis =1)

        def integral_array(MAP_OK_integral,X):
            room = self.room
            return (MAP_OK_integral[min(X[0]+rad,room.H_grid-1),min(X[1]+rad,room.W_grid-1)]
        + MAP_OK_integral[max(X[0]-rad,0),max(X[1]-rad,0)] 
        - MAP_OK_integral[min(X[0]-rad,room.H_grid-1),min(X[1]+rad,room.W_grid-1)] 
        - MAP_OK_integral[min(X[0]+rad,room.H_grid-1),min(X[1]-rad,room.W_grid-1)])

        labels = [integral_array(MAP_OK_integral,X) > 0 for X in det_coordinates.tolist()]
        

        return np.asarray(labels).astype(np.int), np.asarray(labels_body), np.asarray(labels_head)

    def load_batch_train(self,fid,cam,sample_equal = True):
        rois_np,indices_no_null = self.get_rois(fid,cam)
        x = self.get_rgb(fid,cam)
        labels, labels_body, labels_head = self.get_labels(Config.img_index_list[fid],cam,indices_no_null)
        # print 'label unique after get label=',np.unique(labels)
        # print 'labels.shape', labels.shape
        # print 'body shape', labels_body.shape
        # print 'head shape', labels_head.shape 

        #We resample in order to have the same number of positive and negative examples
        if sample_equal:
            n_pos = labels.sum()
            ratio = n_pos*1.0/(labels.shape[0]-n_pos)
            # print 'ratio of pos/neg = ', ratio

            select = []
            for i,lab in enumerate(labels.tolist()):
                
                if lab:
                    select.append(True)
                else:
                    if random.random() < ratio:
                        select.append(True)
                    else:
                        select.append(False)
            # print 'select unique=', np.unique(select)
            rois_np = rois_np[np.array(select)]
            labels = labels[np.array(select)]
            labels_body = labels_body[np.array(select)]
            labels_head = labels_head[np.array(select)]
        # print 'labels.shape', labels.shape
        # print 'body shape', labels_body.shape
        # print 'head shape', labels_head.shape 
        # print 'label unique in load batch=',np.unique(labels)
        return x,rois_np,labels, labels_body, labels_head

    def load_batch_run(self,fid,cam):
        rois_np,indices_no_null = self.get_rois(fid,cam)
        x = self.get_rgb(fid,cam)

        return x,rois_np,indices_no_null


    def visualize_batch(self,x,rois_np ,i = 0,CNN_factor = 4):
        import copy
        rgb = copy.copy(x[i].transpose((1,2,0))) 

        for idbb, bbox in enumerate(rois_np.tolist()[:]):
            color = (2550,0,0)
            bbox = np.asarray(bbox).astype(np.int)
            cv2.rectangle(rgb,(Config.CNN_factor*bbox[1],Config.CNN_factor*bbox[2]),
                          (Config.CNN_factor*bbox[3],Config.CNN_factor*bbox[4]),color,3)

        # plt.imshow(rgb)
        # plt.show()
        return rgb
     
    #draw both GT and estimated orientation
    def visualize_positives(self,x,rois_np,labels,body_labels, head_labels,estimate_rad, i=0, cam=0, CNN_factor = 4):
        import copy
        rgb = copy.copy(x[0].transpose((1,2,0)))
    
        for idbb, bbox in enumerate(rois_np.tolist()[:]):
            # print idbb, bbox
            # print 'score:', labels[idbb]
            color = (255,0,0)
            if labels[idbb][0]>0.5:
                
                bbox = np.asarray(bbox).astype(np.int)
                cv2.rectangle(rgb,(Config.CNN_factor*bbox[1],Config.CNN_factor*bbox[2]),
                              (Config.CNN_factor*bbox[3],Config.CNN_factor*bbox[4]),color,2)
                gp_x = (Config.CNN_factor*bbox[1] + Config.CNN_factor*bbox[3])*0.5
                gp_y = Config.CNN_factor*bbox[4]
                cv2.circle(rgb, (int(gp_x), int(gp_y)), 5, (0,255,0), -2)
                #draw GT orientation
                if body_labels[idbb]<-5:
                    length = 50
                    angle_b = body_labels[idbb]
                    x2_b = int(gp_x + length * math.cos(angle_b))
                    y2_b = int(gp_y + length * math.sin(angle_b))
                    cv2.arrowedLine(rgb, (int(gp_x), int(gp_y)), (x2_b, y2_b), (0, 255, 0), 2)
                    
                    length = 30
                    angle_h = head_labels[idbb]
                    x2_h = int(gp_x + length * math.cos(angle_h))
                    y2_h = int(gp_y + length * math.sin(angle_h))
                    cv2.arrowedLine(rgb, (int(gp_x), int(gp_y)), (x2_h, y2_h), (0, 255, 255), 2)
                    #draw estimated orientation
                    length = 30
                    eh = estimate_rad[idbb,1]#0 body, 1 head
                    x2_eh = int(gp_x + length * math.cos(eh))
                    y2_eh = int(gp_y + length * math.sin(eh))
                    cv2.arrowedLine(rgb, (int(gp_x), int(gp_y)), (x2_eh, y2_eh), (255, 0, 255), 2)
                    
                    length = 50
                    eb = estimate_rad[idbb,0]#0 body, 1 head
                    x2_eb = int(gp_x + length * math.cos(eb))
                    y2_eb = int(gp_y + length * math.sin(eb))
                    # print eh+eb

                    cv2.arrowedLine(rgb, (int(gp_x), int(gp_y)), (x2_eb, y2_eb), (0, 0, 255), 2)
                    # cv2.addWeighted(overlay, 0.5, rgb, 0.5, 0, rgb)
        
        # plt.figure(figsize=(10,10))
        # plt.imshow(rgb)
        # plt.show()
        # plt.imsave('result_orien/cam%d_fid%d.png'%(cam,i), rgb)
        return rgb
    
    #draw estimated orientation with angles input corresponding to RoI
    def visualize_positive_angles(self, x,rois_np,labels,estimate_rad, CNN_factor = 4):
        import copy
        rgb = copy.copy(x[0].transpose((1,2,0))) 
    
        for idbb, bbox in enumerate(rois_np.tolist()[:]):
            color = (255,0,0)
            if labels[idbb][0]>0.5:
                
                bbox = np.asarray(bbox).astype(np.int)
                cv2.rectangle(rgb,(Config.CNN_factor*bbox[1],Config.CNN_factor*bbox[2]),
                              (Config.CNN_factor*bbox[3],Config.CNN_factor*bbox[4]),color,2)
                gp_x = (Config.CNN_factor*bbox[1] + Config.CNN_factor*bbox[3])*0.5
                gp_y = Config.CNN_factor*bbox[4]
                cv2.circle(rgb, (int(gp_x), int(gp_y)), 5, (0,255,0), -2)
                
                #draw estimated orientation
                length = 30
                eh = estimate_rad[idbb,0]#0 body, 1 head
                x2_eh = int(gp_x + length * math.cos(eh))
                y2_eh = int(gp_y + length * math.sin(eh))
                cv2.arrowedLine(rgb, (int(gp_x), int(gp_y)), (x2_eh, y2_eh), (255, 0, 255), 2)

                length = 50
                eb = estimate_rad[idbb,0]#0 body, 1 head
                x2_eb = int(gp_x + length * math.cos(eb))
                y2_eb = int(gp_y + length * math.sin(eb))
                # print eh+eb

                cv2.arrowedLine(rgb, (int(gp_x), int(gp_y)), (x2_eb, y2_eb), (0, 0, 255), 2)
        
        plt.figure(figsize=(20,10))
        plt.imshow(rgb)
        plt.show()
        # plt.imsave('result_orien/cam%d_e55_fid%d.png'%(cam,i), rgb)
        return rgb
    
    #draw estimated orientation with vectors input corresponding to RoI
    def visualize_positivesAndOri(self, x, rois_np, labels , body_labels, head_labels, est_bVec, est_hVec):

        rgb = copy.copy(x[0].transpose((1,2,0))) 
        for idbb, (bbox, bGT, hGT, bEst, hEst) in enumerate( zip(rois_np.tolist()[:], body_labels, head_labels, est_bVec, est_hVec)):
            color = (255,0,0)
            if labels[idbb][0]>0.5:
                bbox = np.asarray(bbox).astype(np.int)
                cv2.rectangle(rgb,(Config.CNN_factor*bbox[1],Config.CNN_factor*bbox[2]),
                              (Config.CNN_factor*bbox[3],Config.CNN_factor*bbox[4]),color,2)
                gp_x = int( (Config.CNN_factor*bbox[1] + Config.CNN_factor*bbox[3])*0.5)
                gp_y = int(Config.CNN_factor*bbox[4])
                cv2.circle(rgb, (gp_x, gp_y), 5, (0,255,0), -2)
                
                #draw estimated orientation
                length = 80
                bGT_x = int(bGT[0]*length)
                bGT_y = int(bGT[1]*length)
                cv2.arrowedLine(rgb, (gp_x,gp_y), (gp_x+bGT_x, gp_y+bGT_y), (0, 255, 0), 2)
                bEst_x = int(bEst[0]*length)
                bEst_y = int(bEst[1]*length)
                cv2.arrowedLine(rgb, (gp_x,gp_y), (gp_x+bEst_x, gp_y+bEst_y), (0, 0, 255), 2)
                
                length = 30
                hGT_x = int(hGT[0]*length)
                hGT_y = int(hGT[1]*length)
                cv2.arrowedLine(rgb, (gp_x,gp_y), (gp_x+hGT_x, gp_y+hGT_y), (255, 255, 0), 2)
                hEst_x = int(hEst[0]*length)
                hEst_y = int(hEst[1]*length)
                cv2.arrowedLine(rgb, (gp_x,gp_y), (gp_x+hEst_x, gp_y+hEst_y), (255, 0, 255), 2)
                
        
        # plt.figure(figsize=(10,10))
        # plt.imshow(rgb)
        # plt.show()
        # plt.imsave('result_orien/cam%d_fid%d.png'%(cam,i), rgb)
        return rgb
        
    # FUNCTIONS TO RUN UNARIES
    #TOFINISH
    
    def run_bulk(self,fid_list = np.arange(len(Config.img_index_list))):
        n_bboxes = self.room.templates_array.shape[1]
        for fid in fid_list:
            print "FID", fid
            scores = np.zeros((self.room.n_cams,n_bboxes)) -10
            for cam in range(self.room.n_cams):
                x,rois_np,indices_no_null= self.load_batch_run(fid,cam)
                p_out_test = self.run_func(x,rois_np)
                scores[cam,indices_no_null] = np.log(p_out_test[:,0])

            np.save(self.unaries_out_path%Config.img_index_list[fid],scores)
            
    def run_test(self,fid = 0, cam =0):
        x,rois_np,l= self.load_batch_run(fid,cam)
        p_out_test = self.run_func(x,rois_np)
        self.visualize_positives(x,rois_np,p_out_test[:,0]>0.8, fid, cam)
        
        
    def run_bulk_features(self,fid_list = np.arange(len(Config.img_index_list)),save_features = True):
        n_bboxes = self.room.templates_array.shape[1]
        for fid in fid_list:
            print "FID", fid
            scores = np.zeros((self.room.n_cams,n_bboxes)) -10
            features = np.zeros((self.room.n_cams,n_bboxes,1024))
            for cam in range(self.room.n_cams):
                x,rois_np,indices_no_null= self.load_batch_run(fid,cam)
                print 'roi', rois_np.shape
                p_out_test = self.run_func(x,rois_np)
                if fid%5==0:
                    self.visualize_positives(x,rois_np,p_out_test[:,0], fid, cam)
                scores[cam,indices_no_null] = np.log(p_out_test[:,0])
                if save_features:
                    x_2_features = self.features_func(x,rois_np)
                    features[cam,indices_no_null,:] = x_2_features

            if not os.path.exists(os.path.dirname(Config.unaries_path)):
                os.makedirs( os.path.dirname(Config.unaries_path) )
            np.save(self.unaries_out_path%Config.img_index_list[fid],scores)
            if save_features:
                np.save(Config.unaries_path_features%Config.img_index_list[fid],features)
    
    # def check_potentials(self, fid, cam, npy_file, mode=1):
    #     n_bboxes = self.room.templates_array.shape[1]
    #     x,rois_np,indices_no_null= self.load_batch_run(fid,cam)
    #     print 'roi shape = ', rois_np.shape,', indices shape = ',indices_no_null.shape
    #     stored_score = np.load(npy_file)
    #     #view the result of single cam
    #     if mode<=1:
    #         unaries_E = -1*stored_score[cam,:]
    #         unaries = unaries_E.clip(0.1,2)*2.0
    #     #view the result of multi cam
    #     else:
    #         unaries_E = -1*stored_score
    #         unaries = unaries_E.clip(0.1,2).min(axis = 0)*2.0
    #     score = np.exp(-1*unaries)
    #     print 'score max = ', score.max(), 'min = ', score.min()
    #     print 'score shape ',score.shape
    #     outImg = self.visualize_positives(x,rois_np,score[indices_no_null], fid, cam)
    #     saveImg = np.zeros((outImg.shape))
    #     saveImg[:,:,0] = outImg[:,:,2]
    #     saveImg[:,:,1] = outImg[:,:,1]
    #     saveImg[:,:,2] = outImg[:,:,0]
    #     # print 'save Img:' + './debug_img/'+npy_file[-10:-3]+'png'
    #     # cv2.imwrite('./debug_img/'+npy_file[-10:-3]+'png',saveImg)

    def run_features(self,fid = 0, cam =0):
        x,rois_np,l= self.load_batch_run(fid,cam)
        x_2_features = self.features_func(x,rois_np)
        return np.asarray(x_2_features)


            
            