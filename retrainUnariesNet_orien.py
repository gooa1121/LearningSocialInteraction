import pickle
import numpy as np
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu2,floatX=float32"
from PIL import Image
import json
from scipy import ndimage
import math
import random

from UnariesNet_orien import unariesNet
import MyConfig_orien as MyConfig


class unaryNet2(unariesNet):
    def __init__(self):
        unariesNet.__init__(self)
        print 'train Img Path = ', MyConfig.trainImgPath
        self.trainImgsPath = MyConfig.trainImgPath
        self.trainLabelsPath = MyConfig.trainLabelPath
        #Path save params
        self.path_save_params = MyConfig.unaries_params_path
        self.train_logs_path = MyConfig.unaries_train_log
        
        self.jsonFile = MyConfig.jsonFile
        # self.imgList = []

    def loadList(self, dataPath, data_ext):
        files = [f for f in os.listdir(dataPath) if os.path.isfile(dataPath + f)]
        files = [i[:-(len(data_ext)+1)] for i in files if i.endswith('.'+data_ext)]
        self.imgList = files

def checkPath(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    unaryNet = unaryNet2()

    unaryNet.loadList(unaryNet.trainImgsPath , 'png')
    checkPath(unaryNet.path_save_params)
    unaryNet.train(0)

if __name__ =="__main__":
    main()