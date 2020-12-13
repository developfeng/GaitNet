import caffe
import numpy as n
import cv2
import os
import pickle as cPickle
import numpy.random as nr
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

def read_im(im_path):
    rim = cv2.imread(im_path)
    inputImage = cv2.resize(rim, (64,64))
    inputImage = n.array(inputImage, dtype=n.float32)
    # substract mean
    inputImage = inputImage -127.5    
    # permute dimensions
    inputImage = inputImage.transpose([2, 0, 1])
    return inputImage

def extract_feature(net,im_dir):
    im_list = n.sort(os.listdir(im_dir))
    num_chn = 6
    num_sample = 20 # Larger number is better, but slower.
    feature = n.zeros((num_sample,256),n.single)
    for sample in xrange(num_sample):
        im0 = []
        im1 = []
        im2 = []
        im3 = []
        im4 = []
        im5 = []
        tmp_frame = nr.randint(len(im_list) - num_chn)
        for chn in xrange(num_chn):
            this_frame = tmp_frame + chn
            im_path = os.path.join(im_dir, im_list[this_frame])
            im_= read_im(im_path)
            if chn == 0:
                im0.append(im_)
            elif chn == 1:
                im1.append(im_)
            elif chn == 2:
                im2.append(im_)
            elif chn == 3:
                im3.append(im_)
            elif chn == 4:
                im4.append(im_)
            elif chn == 5:
                im5.append(im_)

        im0 = n.array(im0).astype(n.float32)
        im1 = n.array(im1).astype(n.float32)
        im2 = n.array(im2).astype(n.float32)        
        im3 = n.array(im3).astype(n.float32)
        im4 = n.array(im4).astype(n.float32)
        im5 = n.array(im5).astype(n.float32)
        net.blobs['im0'].reshape(*im0.shape)
        net.blobs['im0'].data[...] = im0
        net.blobs['im1'].reshape(*im1.shape)
        net.blobs['im1'].data[...] = im1
        net.blobs['im2'].reshape(*im2.shape)
        net.blobs['im2'].data[...] = im2
        net.blobs['im3'].reshape(*im3.shape)
        net.blobs['im3'].data[...] = im3
        net.blobs['im4'].reshape(*im4.shape)
        net.blobs['im4'].data[...] = im4
        net.blobs['im5'].reshape(*im5.shape)
        net.blobs['im5'].data[...] = im5
        net.forward()
        output = n.squeeze(net.blobs['fc_fea'].data)
        feature[sample,:] = output[:]
    feature = n.mean(feature,axis = 0)
    return feature

def get_top_k(preds, g_label, this_p_label,top_k):
    index = n.argsort(-preds,axis=0) #Sort this list with axis-0 and with ascending sequence
    match = 0
    for i in xrange(top_k):
        label_match_g = g_label[index[i]]
        if label_match_g==this_p_label:
            match = 1
    return match
    
def get_acc(probe_fea, probe_lab, gal_fea, gal_lab):
    valid = 0
    top_k = 1#####NOTE!This should be changed if not using top k candicates!
    right_counts = 0
    for p in xrange(len(probe_lab)):
        this_probe_fea = probe_fea[p,:]
        this_p_label = probe_lab[p]
        simlarity = n.zeros(len(gal_lab),dtype = n.single)
        for g in xrange(len(gal_lab)):
            this_gal_fea = gal_fea[g,:]
            simlarity[g] = n.exp(-(((this_probe_fea-this_gal_fea)/256.0)**2).sum())
        this_match = get_top_k(simlarity, gal_lab, this_p_label,top_k)
        right_counts += this_match
        valid +=1
    acc = float(right_counts) / (valid + (valid==0))*100.0
    return acc
    
if __name__ == '__main__':
    pass
    data_path = '/data/SZU/crop-im/'
    model_config = './deploy_gaitnet.prototxt'
    model_data = '../models/szu/gaitnet-sia_iter_30000.caffemodel'
    train_num = 49
    txt_name='./szu_results_tr%d.txt'%train_num
    fall=open(txt_name,"w")
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(model_config, model_data, caffe.TEST)
    
    id_list = n.sort(os.listdir(data_path))

    g_seq_list = ['1','2','3','4']
    p_seq_list = ['5','6','7','8']
    scene_list = ['scene1']
    cloth_list = ['nm']
    
    gait_fea = './feat/szu_feat_tr%d' %train_num
    if not os.path.exists(gait_fea):
        ### Extract features of gallery. 
        condition_list = g_seq_list
        gallery_features = n.zeros((3,len(scene_list)*len(g_seq_list)*(100-train_num),256),n.single)
        gallery_labels = n.zeros((3,len(scene_list)*len(g_seq_list)*(100-train_num)),n.int)
        gallery_valid = n.zeros(3,n.int)
        cont_cloth = -1
        for cloth in cloth_list:
            cont_cloth +=1
            cont_seq = -1
            for id in id_list[train_num:]:
                for con in condition_list:
                    this_dir = os.path.join(data_path, id, con)
                    pic_im_list = n.sort(os.listdir(this_dir))
                    if len(pic_im_list)>10:
                        cont_seq +=1
                        this_fea = extract_feature(net, this_dir)
                        gallery_features[cont_cloth,cont_seq,:] = this_fea[:]
                        gallery_labels[cont_cloth,cont_seq] = int(id) - 1
            gallery_valid[cont_cloth] = cont_seq +1
            print 'Dealing with %s of Gallery!'%cloth
        
        ### Extract features of probe. 
        condition_list = p_seq_list
        probe_features = n.zeros((3,len(scene_list)*len(p_seq_list)*(100-train_num),256),n.single)
        probe_labels = n.zeros((3,len(scene_list)*len(p_seq_list)*(100-train_num)),n.int)
        probe_valid = n.zeros(3,n.int)
        cont_cloth = -1
        for cloth in cloth_list:
            cont_cloth +=1
            cont_seq = -1
            for id in id_list[train_num:]:
                for con in condition_list:
                    this_dir = os.path.join(data_path, id, con)
                    pic_im_list = n.sort(os.listdir(this_dir))
                    if len(pic_im_list)>10:
                        cont_seq +=1
                        this_fea = extract_feature(net, this_dir)
                        probe_features[cont_cloth,cont_seq,:] = this_fea[:]
                        probe_labels[cont_cloth,cont_seq] = int(id) - 1
            probe_valid[cont_cloth] = cont_seq +1
            print 'Dealing with %s of Probe!'%cloth
        dic = {'gallery_features':gallery_features,'gallery_labels':gallery_labels, 'gallery_valid':gallery_valid,'probe_features':probe_features,'probe_labels':probe_labels, 'probe_valid':probe_valid}
        mypickle(gait_fea, dic)
    else:
        dic = myunpickle(gait_fea)
        gallery_features = dic['gallery_features']
        gallery_labels = dic['gallery_labels']
        gallery_valid = dic['gallery_valid']
        probe_features = dic['probe_features']
        probe_labels = dic['probe_labels']
        probe_valid = dic['probe_valid']
    ###Compute the similarity
    fall.write('Prob\Gallery\t')
    for cloth in cloth_list:
        fall.write('%s\t' %cloth)
    for probe in xrange(1):
        fall.write('\n')
        fall.write('%s\t' %cloth_list[probe])
        this_probe_fea = probe_features[probe,:probe_valid[probe],:]
        this_probe_lab = probe_labels[probe,:probe_valid[probe]]
        for gallery in xrange(1):
            this_gal_fea = gallery_features[gallery,:gallery_valid[gallery],:]
            this_gal_lab = gallery_labels[gallery,:gallery_valid[gallery]]
            acc = get_acc(this_probe_fea, this_probe_lab, this_gal_fea, this_gal_lab)
            print 'ACC of Probe = %s and Gallery = %s is %.3f!'%(cloth_list[probe], cloth_list[gallery], acc)
            fall.write('%.3f\t' %acc)
    fall.close()