import argparse
from typing import Dict, List
import os
import numpy as np
from torch.utils.data import DataLoader
from backbone.interface import Interface
from utils.dataset import Dataset
from model import Model
import torch
import xml.etree.ElementTree as ET
import _pickle as cPickle
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_rec(filename):
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        #obj_struct['pose'] = obj.find('pose').text
        #obj_struct['truncated'] = int(obj.find('truncated').text)
        #obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [float(bbox.find('xmin').text),
                              float(bbox.find('ymin').text),
                              float(bbox.find('xmax').text),
                              float(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects


def get_map(backbone_name , path_to_data_dir , checkpoint_path):
    dataset = Dataset(path_to_data_dir, Dataset.Mode.TEST)
    path_to_results_dir = './temp_map'
    evaluator = Evaluator(dataset, path_to_data_dir, path_to_results_dir)

    backbone = Interface.from_name(backbone_name)(pretrained=False)
    model = Model(backbone).to(device)
    print ('*****',checkpoint_path)
    model.load(checkpoint_path)

    label_to_ap_dict = evaluator.evaluate(model)
    mean_ap = np.mean([v for k, v in label_to_ap_dict.items()])
    return mean_ap
def get_ap(rec, prec, use_07_metric=False):

    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
def get_total_ap(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = cPickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        #difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos
        class_recs[imagename] = {'bbox': bbox,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = get_ap(rec, prec, use_07_metric)

    return rec, prec, ap

class Evaluator(object):
    def __init__(self, dataset, path_to_data_dir, path_to_results_dir):
        super().__init__()
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        self._path_to_data_dir = path_to_data_dir
        self._path_to_results_dir = path_to_results_dir
        os.makedirs(self._path_to_results_dir, exist_ok=True)

    def evaluate(self, model):
        all_image_ids, all_detection_bboxes, all_detection_labels, all_detection_probs = [], [], [], []

        with torch.no_grad():
            for batch_index, (image_id_batch, image_batch, scale_batch, _, _) in enumerate(tqdm(self.dataloader)):
                image_id = image_id_batch[0]
                image = image_batch[0].to(device)
                scale = scale_batch[0].to(device).type(torch.FloatTensor)

                forward_input = Model.ForwardInput.Eval(image)
                forward_output = model.eval().forward(forward_input)

                detection_bboxes = forward_output.detection_bboxes.type(torch.FloatTensor) / scale
                detection_labels = forward_output.detection_labels
                detection_probs = forward_output.detection_probs

                all_detection_bboxes.extend(detection_bboxes.tolist())
                all_detection_labels.extend(detection_labels.tolist())
                all_detection_probs.extend(detection_probs.tolist())
                all_image_ids.extend([image_id] * len(detection_bboxes))

        self._write_results(all_image_ids, all_detection_bboxes, all_detection_labels, all_detection_probs)
        path_to_main_dir =  self._path_to_data_dir
        path_to_annotations_dir = os.path.join(self._path_to_data_dir, 'annotation')

        label_to_ap_dict = {}
        for c in range(1, Model.NUM_CLASSES):
            category = Dataset.lab_to_cat_dict[c]
            try:
                _, _, ap = get_total_ap(detpath=os.path.join(self._path_to_results_dir, 'comp3_det_test_{:s}.txt'.format(category)),
                                    annopath=os.path.join(path_to_annotations_dir, '{:s}.xml'),
                                    imagesetfile=os.path.join(path_to_main_dir, 'test.txt'),
                                    classname=category,
                                    cachedir='cache',
                                    ovthresh=0.5,
                                    use_07_metric=True)
            except IndexError:
                ap = 0

            label_to_ap_dict[c] = ap

        return label_to_ap_dict
    def _write_results(self, image_ids: List[str], bboxes: List[List[float]], labels: List[int], probs: List[float]):
        label_to_txt_files_dict = {}
        for c in range(1, Model.NUM_CLASSES):
            label_to_txt_files_dict[c] = open(os.path.join(self._path_to_results_dir, 'comp3_det_test_{:s}.txt'.format(Dataset.lab_to_cat_dict[c])), 'w')

        for image_id, bbox, label, prob in zip(image_ids, bboxes, labels, probs):
            label_to_txt_files_dict[label].write('{:s} {:f} {:f} {:f} {:f} {:f}\n'.format(image_id, prob,
                                                                                          bbox[0], bbox[1], bbox[2], bbox[3]))

        for _, f in label_to_txt_files_dict.items():
            f.close()
