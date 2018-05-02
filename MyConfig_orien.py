datasetPath = '../../../../../cvlabdata2/home/pichen/coco/set2/'
trainImgPath = datasetPath + 'trainImgs/'
trainLabelPath = datasetPath + 'trainLabels/'
jsonFile ='../../../dataset/Coco/json_file/'
downsampleRate = 1

#for training GMM and regression model
log_path = '../Potentials/Parts/log_GMM/'
VGGparam = './VGG/models/paramsBG_noCrowd.pickle'
net_params_path = '../Potentials/Parts/log_GMM/net/'

#for testing GMM
bgParams_path = './VGG/models/paramsBG_noCrowd_e4.pickle'
testImgPath = datasetPath + 'testImgs/'

#for CNN
CNN_factor = 4
n_epochs = 80

#for training UnariesNet
unaries_params_path = '../Unaries_orien_newGT2/trainedModels/'
unaries_train_log = '../Unaries_orien_newGT2/train_unaries.txt'
unaries_test_log = '../Unaries_orien_newGT2/test_unaries.txt'
unaries_boxlist = jsonFile+'trainBoxList.json' # the list of ground truth bounding box
unaries_imgList = jsonFile+'trainImgList.json' # the corresponding list of image name, serves as index to read bbox

unaries_output_path = './Potentials/Unaries_orien1/RunFeatures_transfer/%08d.npy'

#for testing unariesNet
testLabelPath = datasetPath + 'testLabels/'

#for testing
unary_storedParam = '../Unaries_orien_newGT2/trainedModels/params_Unaries_55.pickle'
refinedVGG_storedParam = '../Unaries_orien_newGT2/trainedModels/params_VGG_55.pickle'
#For training to refine bbox localization after training on COCO
# unary_storedParam = '../Unaries/trainedModels_noCrowd/params_Unaries_84.pickle'
# refinedVGG_storedParam = '../Unaries/trainedModels_noCrowd/params_VGG_84.pickle'

#for generate result of body parts and fg
GMM_storedParam = '../Potentials/Parts/log_GMM/net/params_gaussian0_78.pickle'
regress_storedParam = '../Potentials/Parts/log_GMM/net/params_regression0_78.pickle'

# #generate data for unaryNet
# train_part_root = './Potentials/Parts/Run_transfer/coco_pedestrain/train/'
# test_part_root = './Potentials/Parts/Run_transfer/coco_pedestrain/test/'
