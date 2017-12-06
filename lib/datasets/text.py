import os
from datasets.imdb import imdb_text
import datasets.ds_utils as ds_utils
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import re
import cPickle
import subprocess
import uuid
from voc_eval import voc_eval, voc_eval_polygon
from fast_rcnn.config import cfg
from PIL import Image

class ctw1500(imdb_text):
    def __init__(self, dataset):
        imdb_text.__init__(self,dataset['name'])   
        self._label_file = dataset['label_file']
        self._image_root = dataset['image_file']
        self._classes = ('__background__','text') 
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._label_file), \
                '_label_file path does not exist: {}'.format(self._label_file)
        assert os.path.exists(self._image_root), \
                '_image_root does not exist: {}'.format(self._image_root)

        self._labels = self._load_label_image(self._label_file, self._image_root)
        self._image_index = [x for x in range(len(self._labels))]

        self._roidb = self._load_roidb(self._labels)

    def _load_label_image(self, filename, images_path):
        labels = []
        with open(filename.strip(), 'r') as fs, open(images_path.strip(),'r') as pimgs:
            files = fs.readlines()
            imgs_path = pimgs.readlines()
            assert(len(files) == len(imgs_path)), 'image lists and labels list should be the same.'
            for ix, file in enumerate(files):
                assert(os.path.basename(file.strip())[:-4] == os.path.basename(imgs_path[ix].strip())[:-4]), 'label list does not match image list'
                label={}
                label['name'] = file.strip() 
                label['imagePath'] = imgs_path[ix].strip() # image path
                img_tmp = Image.open(label['imagePath'])
                with open(file.strip(), 'r') as f:
                    line_boxes = f.readlines()
                    num_boxes = len(line_boxes)
                    box_info = np.zeros((num_boxes, 4), np.float32) # 4
                    # (xmin, ymin, xmax, ymax)
                    box = np.zeros((num_boxes, 4), np.float32)
                    # (p1_w, p1_h, p2_w, p2_h, ..., pi_w, pi_h, ..., p14_w, p14_h)
                    gt_info = np.zeros((num_boxes, 28), np.float32) # syn
                    for ix, line in enumerate(line_boxes):
                        items = re.split(',', line)
                        # original normalized label (0-1)
                        box_info[ix,:] = [ float(items[i].strip()) for i in range(0,4) ] 

                        box[ix,0] = box_info[ix,0]
                        box[ix,1] = box_info[ix,1]
                        box[ix,2] = box_info[ix,2] # xmax
                        box[ix,3] = box_info[ix,3] # ymax

                        gt_info[ix,:] = [float(items[i].strip()) for i in range(4,32)] # syn

                        assert(int(box[ix,0])>=0), 'xmin should larger than 0 ' + int(box[ix,0])
                        assert(int(box[ix,1])>=0), 'ymin should larger than 0 ' + int(box[ix,1])
                        assert(int(box[ix,2])<=int(img_tmp.size[0])), 'xmax outside image border ' + int(box[ix,2]) + ' '+ str(img_tmp.size[0])
                        assert(int(box[ix,3])<=int(img_tmp.size[1])), 'ymax outside image border' + int(box[ix,3]) + ' '+ str(img_tmp.size[1])
                        assert(int(box[ix,0])< int(box[ix,2])), 'xmin should less than xmax' + int(box[ix,0]) + ' '+ int(box[ix,2])
                        assert(int(box[ix,1])< int(box[ix,3])), 'ymin should less than ymax' + int(box[ix,1]) + ' '+ int(box[ix,3])

                label['gt_boxes'] = box
                label['gt_info'] = gt_info # syn

                labels.append(label)
        print "load images number. {}".format(len(labels))                
        return labels

################## the images' path and names 
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # image_path = os.path.join(self._data_path,'JPEGImages' , 'img_'+index + self._image_ext)
        image_path = os.path.join(self._labels[index]['imagePath'])
        assert os.path.exists(image_path), \
               'Path does not exist: {}'.format(image_path)
        return image_path

################# the annotation's path and name
    def _load_ctw1500_annotation(self, labels, index):
        """
        Load image and bounding boxes info from TXT file in the CTW1500
        format.
        """
        num_objs = len(labels[index]['gt_boxes'])

        gt_boxes = np.zeros((num_objs, 4), dtype = np.uint8)
        gt_classes = np.zeros((num_objs), dtype = np.int32)
        gt_info = np.zeros((num_objs,28) , dtype = np.int32)

        gt_boxes = labels[index]['gt_boxes']
        gt_info = labels[index]['gt_info'] # syn
        # 
        gt_name = labels[index]['imagePath']
        cls = self._class_to_ind['text']

        gt_classes[:] = cls

        return {'boxes' : gt_boxes,
                'gt_classes': gt_classes,
                'gt_info': gt_info, # syn
                'flipped' : False,
                'imagePath' : gt_name}

    def _load_roidb(self, labels):
        cache_file = os.path.join(self.cache_path, self.name+'_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb =cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        num = len(labels)
        print 'load sample number = ', num
        gt_roidb = [ self._load_ctw1500_annotation(labels, i) for i in range(num) ]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

############################# the path of results 
    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + 'text' + '_{:s}.txt'
        path = os.path.join(
            self.output_dir,       
            filename)
        return path

#############################the detection result of test will be writen in results folder in txt
    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = self._get_voc_results_file_template().format(cls)
            print(filename)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self._image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        # a=input('check here')
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(str(index)   , dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _qua_write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = self._get_voc_results_file_template().format(cls)
            print(filename)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self._image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(str(index)   , dets[k, 4],
                                       dets[k, 0] + 1 + dets[k, 5], dets[k, 1] + 1 + dets[k, 6],
                                       dets[k, 0] + 1 + dets[k, 7], dets[k, 1] + 1 + dets[k, 8],
                                       dets[k, 0] + 1 + dets[k, 9], dets[k, 1] + 1 + dets[k, 10],
                                       dets[k, 0] + 1 + dets[k, 11], dets[k, 1] + 1 + dets[k,12]
                                       ))                        

    def _curve_write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = self._get_voc_results_file_template().format(cls)
            print(filename)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self._image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in xrange(dets.shape[0]):
                        info_bbox = dets[k, 5:33] # indexing
                        pts = [info_bbox[i] for i in xrange(28)]
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(str(index)   , dets[k, 4],
                                       dets[k, 0] + pts[0], dets[k, 1] + pts[1],
                                       dets[k, 0] + pts[2], dets[k, 1] + pts[3],dets[k, 0] + pts[4], dets[k, 1] + pts[5],
                                       dets[k, 0] + pts[6], dets[k, 1] + pts[7],dets[k, 0] + pts[8], dets[k, 1] + pts[9],dets[k, 0] + pts[10], dets[k, 1] + pts[11],
                                       dets[k, 0] + pts[12], dets[k, 1] + pts[13],
                                       dets[k, 0] + pts[14], dets[k, 1] + pts[15],
                                       dets[k, 0] + pts[16], dets[k, 1] + pts[17],dets[k, 0] + pts[18], dets[k, 1] + pts[19],
                                       dets[k, 0] + pts[20], dets[k, 1] + pts[21],dets[k, 0] + pts[22], dets[k, 1] + pts[23],dets[k, 0] + pts[24], dets[k, 1] + pts[25],
                                       dets[k, 0] + pts[26], dets[k, 1] + pts[27]
                                       ))                   

#########################call voc_eval to evaluate the rec and prec ,mAP
    def _do_python_eval(self, output_dir = 'output'):
        cachedir = os.path.join(self.output_dir)
        aps = []
        # The PASCAL VOC metric changed in 2010
        # use_07_metric = True if int(self._year) < 2010 else False
        use_07_metric = False
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            # rec, prec, ap = voc_eval(
                # filename, self._label_file, self._image_root, cls, cachedir, ovthresh=0.5,
                # use_07_metric=use_07_metric)
            rec, prec, ap = voc_eval_polygon(
                filename, self._label_file, self._image_root, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            F=2.0/(1/rec[-1]+1/prec[-1])
            # print F
            if not os.path.isdir('results'):
                os.mkdir('results')
            f=open('results/test_result.txt','a')
            f.writelines('rec:%.3f prec:%.3f F-measure:%.3f \n\n'% (rec[-1],prec[-1],F))
            f.close()
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        # for ap in aps:
            # print('{:.3f}'.format(ap))
        print 'rec:%%%%%%%%%%%%'
        print rec[-1]
        print 'prec:###########'           
        print prec[-1]
        print 'F-measure'
        print F
        # print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')

    def evaluate_detections(self, all_boxes, output_dir):
        self.output_dir = output_dir
        self._curve_write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)


    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
        else:
            self.config['use_salt'] = True
