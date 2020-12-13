"""
For GaitNet data layer

by Chunfeng Song
"""

import caffe
import numpy as np
import yaml
from random import shuffle
import numpy.random as nr
import cv2
import os
import pickle as cPickle
import pdb

def mypickle(filename, data):
    fo = open(filename, "wb")
    cPickle.dump(data, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    fo.close()
    
def myunpickle(filename):
    if not os.path.exists(filename):
        raise UnpickleError("Path '%s' does not exist." % filename)
    fo = open(filename, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def szu_processor(train_num, im_path, gt_path):
    szu_gait_dic = './szu_gait_dic'
    if not os.path.exists(szu_gait_dic):
        im_list = []
        gt_list = []
        labels = []
        id_list = np.sort(os.listdir(im_path))
        idx = -1
        for id in id_list[:train_num]:
            idx +=1
            condition_list = np.sort(os.listdir(os.path.join(im_path, id)))
            for con in condition_list:
                pic_im_list = np.sort(os.listdir(os.path.join(im_path, id, con)))
                pic_gt_list = np.sort(os.listdir(os.path.join(gt_path, id, con)))
                if len(pic_im_list)>10 and len(pic_im_list)==len(pic_gt_list):
                    im_dir = os.path.join(im_path, id, con)
                    gt_dir = os.path.join(gt_path, id, con)
                    im_list.append(im_dir)
                    gt_list.append(gt_dir)
                    labels.append(idx)
        dic = {'im_list':im_list,'gt_list':gt_list, 'labels':labels}
        mypickle(szu_gait_dic, dic, compress=False)
    # load the saved data (to resume)
    else:
        dic = myunpickle(szu_gait_dic)
        im_list = dic['im_list']
        gt_list = dic['gt_list']
        labels = dic['labels']
    return labels, im_list, gt_list

def load_data(im_path, gt_path, width, height):
    """
    Load input image and preprocess for Caffe:
    - cast to float
    - switch channels RGB -> BGR
    - subtract mean
    - transpose to channel x height x width order
    """

    oim = cv2.imread(im_path)
    if not os.path.exists(gt_path):
        gt_im = np.zeros((oim.shape[0],oim.shape[1]), dtype=np.uint8)
    else:
        gt_im= cv2.cvtColor(cv2.imread(gt_path),cv2.COLOR_BGR2GRAY)

    inputImage = cv2.resize(oim, (width, height))
    inputImage = np.array(inputImage, dtype=np.float32)
    
    # substract mean
    inputImage = inputImage -127.5
    
    # permute dimensions
    inputImage = inputImage.transpose([2, 0, 1])
    
    ###GT
    inputGt = np.array(cv2.resize(gt_im, (width, height)), dtype=np.float32)
    inputGt = inputGt/255.0
    inputGt = inputGt[np.newaxis, ...]
    return inputImage, inputGt

class GaitNet_Seg(caffe.Layer):
    """Data layer for training"""   
    def setup(self, bottom, top): 
        self.width = 64
        self.height = 64
        layer_params = yaml.load(self.param_str)
        self.train_num = layer_params['train_num']
        self.batch_size = layer_params['batch_size']
        self.im_path = layer_params['im_path']
        self.gt_path = layer_params['gt_path']
        self.mode = layer_params['mode']
        self.labels, self.im_list, self.gt_list = szu_processor(self.train_num, self.im_path, self.gt_path)
        self.idx = 0
        self.num_chn = 6
        self.data_num = len(self.im_list) # num of data pairs
        self.rnd_list = np.arange(self.data_num ) # random shuffle the images list
        shuffle(self.rnd_list)
        
    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.im0
        top[1].data[...] = self.im1
        top[2].data[...] = self.im2
        top[3].data[...] = self.im3
        top[4].data[...] = self.im4
        top[5].data[...] = self.im5
        top[6].data[...] = self.gt0
        top[7].data[...] = self.gt1
        top[8].data[...] = self.gt2
        top[9].data[...] = self.gt3
        top[10].data[...] = self.gt4
        top[11].data[...] = self.gt5

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        # load image + label pairs
        self.im0 = []
        self.im1 = []
        self.im2 = []
        self.im3 = []
        self.im4 = []
        self.im5 = []
        self.gt0 = []
        self.gt1 = []
        self.gt2 = []
        self.gt3 = []
        self.gt4 = []
        self.gt5 = []

        for i in xrange(self.batch_size):
            if self.idx != self.data_num:
                cur_idx = self.rnd_list[self.idx]
            else:
                self.idx = 0
                cur_idx = self.rnd_list[self.idx]
            im_dir = self.im_list[cur_idx]
            gt_dir = self.gt_list[cur_idx]
            im_list = np.sort(os.listdir(im_dir))
            tmp_frame = nr.randint(len(im_list) - self.num_chn)
            for chn in xrange(self.num_chn):
                this_frame = tmp_frame + chn
                im_path = os.path.join(im_dir, im_list[this_frame])
                gt_path = os.path.join(gt_dir, im_list[this_frame])
                im_, gt_= load_data(im_path, gt_path, self.width, self.height)
                if chn == 0:
                    self.im0.append(im_)
                    self.gt0.append(gt_)
                elif chn == 1:
                    self.im1.append(im_)
                    self.gt1.append(gt_)
                elif chn == 2:
                    self.im2.append(im_)
                    self.gt2.append(gt_)
                elif chn == 3:
                    self.im3.append(im_)
                    self.gt3.append(gt_)
                elif chn == 4:
                    self.im4.append(im_)
                    self.gt4.append(gt_)
                elif chn == 5:
                    self.im5.append(im_)
                    self.gt5.append(gt_)
            self.idx +=1

        self.im0 = np.array(self.im0).astype(np.float32)
        self.im1 = np.array(self.im1).astype(np.float32)
        self.im2 = np.array(self.im2).astype(np.float32)
        self.im3 = np.array(self.im3).astype(np.float32)
        self.im4 = np.array(self.im4).astype(np.float32)
        self.im5 = np.array(self.im5).astype(np.float32)
        self.gt0 = np.array(self.gt0).astype(np.float32)
        self.gt1 = np.array(self.gt1).astype(np.float32)
        self.gt2 = np.array(self.gt2).astype(np.float32)
        self.gt3 = np.array(self.gt3).astype(np.float32)
        self.gt4 = np.array(self.gt4).astype(np.float32)
        self.gt5 = np.array(self.gt5).astype(np.float32)
        
        # reshape tops to fit blobs
        top[0].reshape(*self.im0.shape)
        top[1].reshape(*self.im1.shape)
        top[2].reshape(*self.im2.shape)
        top[3].reshape(*self.im3.shape)
        top[4].reshape(*self.im4.shape)
        top[5].reshape(*self.im5.shape)
        
        top[6].reshape(*self.gt0.shape)
        top[7].reshape(*self.gt1.shape)
        top[8].reshape(*self.gt2.shape)
        top[9].reshape(*self.gt3.shape)
        top[10].reshape(*self.gt4.shape)
        top[11].reshape(*self.gt5.shape)


class GaitNet(caffe.Layer):
    """Data layer for training"""   
    def setup(self, bottom, top): 
        self.width = 64
        self.height = 64
        layer_params = yaml.load(self.param_str)
        self.train_num = layer_params['train_num']
        self.batch_size = layer_params['batch_size']
        self.im_path = layer_params['im_path']
        self.gt_path = layer_params['gt_path']
        self.mode = layer_params['mode']
        self.labels, self.im_list, self.gt_list = szu_processor(self.train_num, self.im_path, self.gt_path)
        self.idx = 0
        self.num_chn = 6
        self.data_num = len(self.im_list) # num of data pairs
        self.rnd_list = np.arange(self.data_num ) # random shuffle the images list
        shuffle(self.rnd_list)
        
    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.im0
        top[1].data[...] = self.im1
        top[2].data[...] = self.im2
        top[3].data[...] = self.im3
        top[4].data[...] = self.im4
        top[5].data[...] = self.im5
        top[6].data[...] = self.gt0
        top[7].data[...] = self.gt1
        top[8].data[...] = self.gt2
        top[9].data[...] = self.gt3
        top[10].data[...] = self.gt4
        top[11].data[...] = self.gt5
        top[12].data[...] = self.label

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        # load image + label pairs
        self.im0 = []
        self.im1 = []
        self.im2 = []
        self.im3 = []
        self.im4 = []
        self.im5 = []
        self.gt0 = []
        self.gt1 = []
        self.gt2 = []
        self.gt3 = []
        self.gt4 = []
        self.gt5 = []
        self.label = []

        for i in xrange(self.batch_size):
            if self.idx != self.data_num:
                cur_idx = self.rnd_list[self.idx]
            else:
                self.idx = 0
                cur_idx = self.rnd_list[self.idx]
            im_dir = self.im_list[cur_idx]
            gt_dir = self.gt_list[cur_idx]
            im_list = np.sort(os.listdir(im_dir))
            tmp_frame = nr.randint(len(im_list) - self.num_chn)
            for chn in xrange(self.num_chn):
                this_frame = tmp_frame + chn
                im_path = os.path.join(im_dir, im_list[this_frame])
                gt_path = os.path.join(gt_dir, im_list[this_frame])
                im_, gt_= load_data(im_path, gt_path, self.width, self.height)
                if chn == 0:
                    self.im0.append(im_)
                    self.gt0.append(gt_)
                elif chn == 1:
                    self.im1.append(im_)
                    self.gt1.append(gt_)
                elif chn == 2:
                    self.im2.append(im_)
                    self.gt2.append(gt_)
                elif chn == 3:
                    self.im3.append(im_)
                    self.gt3.append(gt_)
                elif chn == 4:
                    self.im4.append(im_)
                    self.gt4.append(gt_)
                elif chn == 5:
                    self.im5.append(im_)
                    self.gt5.append(gt_)
            self.label.append (self.labels[cur_idx])
            self.idx +=1

        self.im0 = np.array(self.im0).astype(np.float32)
        self.im1 = np.array(self.im1).astype(np.float32)
        self.im2 = np.array(self.im2).astype(np.float32)
        self.im3 = np.array(self.im3).astype(np.float32)
        self.im4 = np.array(self.im4).astype(np.float32)
        self.im5 = np.array(self.im5).astype(np.float32)
        self.gt0 = np.array(self.gt0).astype(np.float32)
        self.gt1 = np.array(self.gt1).astype(np.float32)
        self.gt2 = np.array(self.gt2).astype(np.float32)
        self.gt3 = np.array(self.gt3).astype(np.float32)
        self.gt4 = np.array(self.gt4).astype(np.float32)
        self.gt5 = np.array(self.gt5).astype(np.float32)
        self.label = np.array(self.label).astype(np.float32)
        
        # reshape tops to fit blobs
        top[0].reshape(*self.im0.shape)
        top[1].reshape(*self.im1.shape)
        top[2].reshape(*self.im2.shape)
        top[3].reshape(*self.im3.shape)
        top[4].reshape(*self.im4.shape)
        top[5].reshape(*self.im5.shape)
        
        top[6].reshape(*self.gt0.shape)
        top[7].reshape(*self.gt1.shape)
        top[8].reshape(*self.gt2.shape)
        top[9].reshape(*self.gt3.shape)
        top[10].reshape(*self.gt4.shape)
        top[11].reshape(*self.gt5.shape)
        top[12].reshape(*self.label.shape)

            
class GaitNet_Sia(caffe.Layer):
    """Data layer for training"""   
    def setup(self, bottom, top): 
        self.width = 64
        self.height = 64
        layer_params = yaml.load(self.param_str)
        self.train_num = layer_params['train_num']
        self.batch_size = layer_params['batch_size']
        self.pos_pair_mining = True
        self.pos_pair_num = int(0.3*self.batch_size)# 30% pos pair
        self.im_path = layer_params['im_path']
        self.gt_path = layer_params['gt_path']
        self.mode = layer_params['mode']
        self.labels, self.im_list, self.gt_list = szu_processor(self.train_num, self.im_path, self.gt_path)
        self.idx = 0
        self.test = 0
        self.num_chn = 6
        self.data_num = len(self.im_list) # num of data pairs
        self.rnd_list = np.arange(self.data_num ) # random the images list
        shuffle(self.rnd_list)
        
    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.im0
        top[1].data[...] = self.im1
        top[2].data[...] = self.im2
        top[3].data[...] = self.im3
        top[4].data[...] = self.im4
        top[5].data[...] = self.im5
        top[6].data[...] = self.gt0
        top[7].data[...] = self.gt1
        top[8].data[...] = self.gt2
        top[9].data[...] = self.gt3
        top[10].data[...] = self.gt4
        top[11].data[...] = self.gt5
        top[12].data[...] = self.label
        top[13].data[...] = self.siam_label

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        # load image + label image pairs
        self.im0 = []
        self.im1 = []
        self.im2 = []
        self.im3 = []
        self.im4 = []
        self.im5 = []
        self.gt0 = []
        self.gt1 = []
        self.gt2 = []
        self.gt3 = []
        self.gt4 = []
        self.gt5 = []
        self.label = []
        self.siam_label = []

        for i in xrange(self.batch_size):
            if self.idx != self.data_num:
                cur_idx = self.rnd_list[self.idx]
            else:
                self.idx = 0
                cur_idx = self.rnd_list[self.idx]
            im_dir = self.im_list[cur_idx]
            gt_dir = self.gt_list[cur_idx]
            im_list = np.sort(os.listdir(im_dir))
            tmp_frame = nr.randint(len(im_list) - self.num_chn)
            for chn in xrange(self.num_chn):
                this_frame = tmp_frame + chn
                im_path = os.path.join(im_dir, im_list[this_frame])
                gt_path = os.path.join(gt_dir, im_list[this_frame])
                im_, gt_= load_data(im_path, gt_path, self.width, self.height)
                if chn == 0:
                    self.im0.append(im_)
                    self.gt0.append(gt_)
                elif chn == 1:
                    self.im1.append(im_)
                    self.gt1.append(gt_)
                elif chn == 2:
                    self.im2.append(im_)
                    self.gt2.append(gt_)
                elif chn == 3:
                    self.im3.append(im_)
                    self.gt3.append(gt_)
                elif chn == 4:
                    self.im4.append(im_)
                    self.gt4.append(gt_)
                elif chn == 5:
                    self.im5.append(im_)
                    self.gt5.append(gt_)
            self.label.append(self.labels[cur_idx])
            self.idx +=1
        if self.pos_pair_mining:
            for i in xrange(self.batch_size):
                if i > self.pos_pair_num:
                    if self.idx != self.data_num:
                        cur_idx = self.rnd_list[self.idx]
                    else:
                        self.idx = 0
                        cur_idx = self.rnd_list[self.idx]
                    self.idx +=1
                    im_dir = self.im_list[cur_idx]
                    gt_dir = self.gt_list[cur_idx]
                    label = self.labels[cur_idx]
                    if label==self.label[i]:
                        self.siam_label.append(int(1))#Also will get pos pairs, maybe few.
                    else:
                        self.siam_label.append(int(0))#neg pair
                else:
                    im_dir, gt_dir = self.pair_mining(self.label[i])
                    label = self.label[i]
                    self.siam_label.append(int(1))# pos pair
                im_list = np.sort(os.listdir(im_dir))
                tmp_frame = nr.randint(len(im_list) - self.num_chn)
                for chn in xrange(self.num_chn):
                    this_frame = tmp_frame + chn
                    im_path = os.path.join(im_dir, im_list[this_frame])
                    gt_path = os.path.join(gt_dir, im_list[this_frame])
                    im_, gt_= load_data(im_path, gt_path, self.width, self.height)
                    if chn == 0:
                        self.im0.append(im_)
                        self.gt0.append(gt_)
                    elif chn == 1:
                        self.im1.append(im_)
                        self.gt1.append(gt_)
                    elif chn == 2:
                        self.im2.append(im_)
                        self.gt2.append(gt_)
                    elif chn == 3:
                        self.im3.append(im_)
                        self.gt3.append(gt_)
                    elif chn == 4:
                        self.im4.append(im_)
                        self.gt4.append(gt_)
                    elif chn == 5:
                        self.im5.append(im_)
                        self.gt5.append(gt_)
                self.label.append(label)    
                
        self.im0 = np.array(self.im0).astype(np.float32)
        self.im1 = np.array(self.im1).astype(np.float32)
        self.im2 = np.array(self.im2).astype(np.float32)
        self.im3 = np.array(self.im3).astype(np.float32)
        self.im4 = np.array(self.im4).astype(np.float32)
        self.im5 = np.array(self.im5).astype(np.float32)
        self.gt0 = np.array(self.gt0).astype(np.float32)
        self.gt1 = np.array(self.gt1).astype(np.float32)
        self.gt2 = np.array(self.gt2).astype(np.float32)
        self.gt3 = np.array(self.gt3).astype(np.float32)
        self.gt4 = np.array(self.gt4).astype(np.float32)
        self.gt5 = np.array(self.gt5).astype(np.float32)
        self.label = np.array(self.label).astype(np.float32)
        self.siam_label = np.array(self.siam_label).astype(np.float32)

        
        top[0].reshape(*self.im0.shape)
        top[1].reshape(*self.im1.shape)
        top[2].reshape(*self.im2.shape)
        top[3].reshape(*self.im3.shape)
        top[4].reshape(*self.im4.shape)
        top[5].reshape(*self.im5.shape)
        
        top[6].reshape(*self.gt0.shape)
        top[7].reshape(*self.gt1.shape)
        top[8].reshape(*self.gt2.shape)
        top[9].reshape(*self.gt3.shape)
        top[10].reshape(*self.gt4.shape)
        top[11].reshape(*self.gt5.shape)
        top[12].reshape(*self.label.shape)
        top[13].reshape(*self.siam_label.shape)
        
            
    def pair_mining(self, label):
        """
        Mining postive pairs for siamese network training.
        """
        id_list = np.sort(os.listdir(self.im_path))
        condition_list = np.sort(os.listdir(os.path.join(self.im_path, id_list[label])))

        not_get = True
        attempts = 0
        while(not_get):
            attempts +=1
            if attempts>=100:
                print '[ERROR!] There is not such a label %d !!!!!!' %label
                break
            select_seq = nr.randint(len(condition_list))
            im_dir = os.path.join(self.im_path, id_list[label], condition_list[select_seq])
            gt_dir = os.path.join(self.gt_path, id_list[label], condition_list[select_seq])
            if os.path.exists(im_dir) and os.path.exists(gt_dir):
                pic_im_list = np.sort(os.listdir(im_dir))
                if len(pic_im_list)>10:
                    not_get = False
                    return im_dir, gt_dir
                    
