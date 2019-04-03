import argparse
import os
import time
from collections import deque
import pickle as pkl
import torch
import xml.etree.ElementTree as ET
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from backbone.interface import Interface
from dataset import Dataset
from model import Model
from utils.bbox import BBox
from evaluate import get_map_score
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def make_predict_annotation(checkpoint_model, backbone_name, predicted_dir):
    backbone = Interface.from_name(backbone_name)(pretrained=False)
    #model = Model(backbone).cuda()
    model = Model(backbone).to(device)
    model.load(checkpoint_model)
    inp_img_dir = './data/images/'
    #forward_input = Model.ForwardInput.Eval(image_tensor.cuda())
    with open('data/test.txt','r') as files:
        lines = files.readlines()
        val_img_files = [line.rstrip()+'.png' for line in lines]
    
    for file in tqdm(val_img_files):
        if file.endswith('.png'):
            file_name = file.split('.')[0]
            inp_img = os.path.join(inp_img_dir,file)
            out_txt = os.path.join(predicted_dir,file_name+'.txt')
            #print(out_txt)
            image = transforms.Image.open(inp_img)
            image_tensor, scale = Dataset.preprocess(image)
            image_tensor = image_tensor.to(device)
            forward_input = Model.ForwardInput.Eval(image_tensor)
            forward_output= model.eval().forward(forward_input)

            detection_bboxes = forward_output.detection_bboxes / scale
            detection_labels = forward_output.detection_labels
            detection_probs = forward_output.detection_probs

            for bbox, label, prob in zip(detection_bboxes.tolist(), detection_labels.tolist(), detection_probs.tolist()):
                #if prob < 0.5:
                    #continue
                bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
                category = Dataset.lab_to_cat_dict[label]
                txt_val = "\n{} {} {} {} {} {}".format(category, prob, bbox.left, bbox.top, bbox.right, bbox.bottom)
                with open(out_txt,'a') as out_file:
                    out_file.write(txt_val)
def _train(backbone_name, path_to_data_dir, path_to_checkpoints_dir):
    dataset = Dataset(path_to_data_dir, mode=Dataset.Mode.TRAIN)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

    backbone = Interface.from_name(backbone_name)(pretrained=True)
    model = Model(backbone).to(device)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0005)
    scheduler = StepLR(optimizer, step_size=50000, gamma=0.1)
    ground_truth_dir = 'val_dir/ground-truth/'
    if not os.path.exists(ground_truth_dir):
        os.makedirs(ground_truth_dir)
    predicted_dir = 'val_dir/predicted/'
    if not os.path.exists(predicted_dir):
        os.makedirs(predicted_dir)
    step = 0
    time_checkpoint = time.time()
    losses = deque(maxlen=100)
    should_stop = False
 
    num_steps_to_display = 10
    num_steps_to_snapshot = 1000
    num_steps_to_stop_training = 80000
    print ('Creating Ground Truth Annotations for validation')
    with open(os.path.join(path_to_data_dir, 'test.txt'),'r') as files:
        lines = files.readlines()
        val_img_files = [line.rstrip() for line in lines]
    annotations = pkl.load(open(os.path.join(path_to_data_dir, 'annt/annotation_file.pkl'), 'rb'))
    print('Start training')

    while not should_stop:
        for batch_index, (_, image_batch, _, bboxes_batch, labels_batch) in enumerate(dataloader):
            #assert image_batch.shape[0] == 1, 'only batch size of 1 is supported'

            image = image_batch[0].to(device)
            bboxes = bboxes_batch[0].to(device)
            labels = labels_batch[0].to(device)
            forward_input = Model.ForwardInput.Train(image, gt_classes=labels, gt_bboxes=bboxes)
            forward_output= model.train().forward(forward_input)

            loss = forward_output.anchor_objectness_loss + forward_output.anchor_transformer_loss + \
                forward_output.proposal_class_loss + forward_output.proposal_transformer_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            step += 1
            with open(os.path.join(path_to_checkpoints_dir,'loss_log.txt'),'a') as logfile:
                    to_write = "step: {}, loss:{}\n".format(step, loss)
                    logfile.write(to_write)
            '''
            if step % 40 == 0:
                map_score = get_map('resnet101', './data')
                with open(os.path.join(path_to_checkpoints_dir,'map_log.txt'),'a') as maplog:
                    to_write = "step: {}, loss:{}, map:{} \n".format(step, loss, map_score)
                    maplog.write(to_write)
                print ('***The MAP score is {}***'.format(map_score))
            '''
            if step % num_steps_to_display == 0:
                elapsed_time = time.time() - time_checkpoint
                time_checkpoint = time.time()
                steps_per_sec = num_steps_to_display / elapsed_time
                avg_loss = sum(losses) / len(losses)
                lr = scheduler.get_lr()[0]
                print('[Step {}] Avg. Loss = {}, Learning Rate = {} ({} steps/sec)'.format(step, avg_loss,lr, steps_per_sec))

            
            if step % num_steps_to_snapshot == 0:
                
                path_to_checkpoint = model.save(path_to_checkpoints_dir, step)
                make_predict_annotation(path_to_checkpoint,backbone_name, predicted_dir)
                map_score = get_map_score()
                with open(os.path.join(path_to_checkpoints_dir,'map_log.txt'),'a') as maplog:
                    to_write = "step: {}, loss:{}, map:{} \n".format(step, loss, map_score)
                    maplog.write(to_write)
                print ('***The MAP score is {}***'.format(map_score))
                print('Model saved to {}'.format(path_to_checkpoint))

            if step == num_steps_to_stop_training:
                should_stop = True
                break

    print('Done')


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('-b', '--backbone', choices=['vgg16', 'resnet101'], required=True, help='name of backbone model')
        parser.add_argument('-d', '--data_dir', default='./data', help='path to data directory')
        parser.add_argument('-c', '--checkpoints_dir', default='./checkpoints', help='path to checkpoints directory')
        args = parser.parse_args()

        backbone_name = args.backbone
        path_to_data_dir = args.data_dir
        path_to_checkpoints_dir = args.checkpoints_dir

        os.makedirs(path_to_checkpoints_dir, exist_ok=True)

        _train(backbone_name, path_to_data_dir, path_to_checkpoints_dir)

    main()
