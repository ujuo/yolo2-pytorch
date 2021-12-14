import os
import torch
import datetime

from darknet import Darknet19
from loss import YoloLoss

#from datasets.pascal_voc import VOCDataset
from datasets.yolov2_tiny import VOCDataset
import utils.yolo as yolo_utils
import utils.network as net_utils
from utils.timer import Timer
import cfgs.config as cfg
from random import randint


try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None


# data loader
imdb = VOCDataset(cfg.imdb_train, cfg.DATA_DIR, cfg.train_batch_size,
                  yolo_utils.preprocess_train, processes=2, shuffle=True,
                  dst_size=cfg.multi_scale_inp_size)
# dst_size=cfg.inp_size)
print('load data succ...')
#print("trained_model : ", cfg.trained_model)


net = Darknet19()

net.cuda()

use_trained_model = 1
#use_trained_model = os.path.join(cfg.train_output_dir,'darknet19_trainval_224_12680+310.pth')
use_trained_model = os.path.join(cfg.train_output_dir,'best_weights.pth')

if use_trained_model:
    print("===========use_trained_model")
    lr = cfg.init_learning_rate
    optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                        momentum=cfg.momentum,
                                        weight_decay=cfg.weight_decay)
    
    checkpoint = torch.load(use_trained_model)
    net.load_state_dict(checkpoint['state'])
    optimizer.load_state_dict(checkpoint['optim'])
    checkpoint_epoch = checkpoint['epoch']



#net_utils.load_net("yolov2-tiny-custom-224.pth",net)
print(net)
# pretrained_model = cfg.trained_model
# net_utils.load_net(pretrained_model, net)
# net.load_from_npz(cfg.pretrained_model, num_conv=18)
# net.cuda()
net.train()
print('load net succ...')

if use_trained_model:
    print("************use trained model")
    start_epoch = checkpoint_epoch
    print("start_epoch: ", start_epoch)
else: # optimizer
    start_epoch = 0
    lr = cfg.init_learning_rate
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=cfg.momentum,
                            weight_decay=cfg.weight_decay)

# tensorboad
use_tensorboard = cfg.use_tensorboard and SummaryWriter is not None
# use_tensorboard = False
if use_tensorboard:
    summary_writer = SummaryWriter(os.path.join(cfg.TRAIN_DIR, 'runs', cfg.exp_name))
else:
    summary_writer = None

best_weights = 0
batch_per_epoch = imdb.batch_per_epoch
train_loss = 0
bbox_loss, iou_loss, cls_loss = 0., 0., 0.
cnt = 0
t = Timer()
step_cnt = 0
size_index = 0

yolov2 = YoloLoss()

##print("start_epoch: ", start_epoch, "imdb.batch_per_epoch: ", imdb.batch_per_epoch)
##print("cfg.max_epoch: ", cfg.max_epoch)
##print("range: ", start_epoch*imdb.batch_per_epoch, " - ",cfg.max_epoch*imdb.batch_per_epoch)
for step in range(start_epoch * imdb.batch_per_epoch,
                  cfg.max_epoch * imdb.batch_per_epoch):
    t.tic()
    # batch
    batch = imdb.next_batch(size_index)
    im = batch['images']
    gt_boxes = batch['gt_boxes']
    gt_classes = batch['gt_classes']
    dontcare = batch['dontcare']
    orgin_im = batch['origin_im']

    ##print("batch len: " ,len(batch))
    ##print("im shape: ", im.shape)
    # forward
    im_data = net_utils.np_to_variable(im,
                                       is_cuda=True,
                                       volatile=False).permute(0, 3, 1, 2)

    ##print("im_data shape: ", im_data.shape) #(16,224,224,3)
    ##print("im_data stride: ", im_data.stride())
###loss 추가 수정
    #box_pred, iou_pred, prob_pred = net(im_data, gt_boxes, gt_classes, dontcare, size_index)
    out = net(im_data)
    box_pred, iou_pred, prob_pred = yolov2(out,gt_boxes, gt_classes, dontcare, size_index)
###
    ##print("+++++++++++++++++++++++++++++++++++++++")
    # backward
    #loss = net.loss
    loss = yolov2.loss
    bbox_loss += yolov2.bbox_loss.data.cpu().numpy()
    iou_loss += yolov2.iou_loss.data.cpu().numpy()
    cls_loss += yolov2.cls_loss.data.cpu().numpy()
    train_loss += loss.data.cpu().numpy()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    cnt += 1
    step_cnt += 1
    duration = t.toc()
    if step % cfg.disp_interval == 0:
        train_loss /= cnt
        bbox_loss /= cnt
        iou_loss /= cnt
        cls_loss /= cnt
        if cfg.best_weights > train_loss:
            cfg.best_weights = train_loss
            best_weights = 1
        print(('epoch %d[%d/%d], loss: %.3f, bbox_loss: %.3f, iou_loss: %.3f, '
               'cls_loss: %.3f (%.2f s/batch, rest:%s)' %
               (imdb.epoch, step_cnt, batch_per_epoch, train_loss, bbox_loss,
                iou_loss, cls_loss, duration,
                str(datetime.timedelta(seconds=int((batch_per_epoch - step_cnt) * duration))))))  # noqa
        

        if summary_writer and step % cfg.log_interval == 0:
            summary_writer.add_scalar('loss_train', train_loss, step)
            summary_writer.add_scalar('loss_bbox', bbox_loss, step)
            summary_writer.add_scalar('loss_iou', iou_loss, step)
            summary_writer.add_scalar('loss_cls', cls_loss, step)
            summary_writer.add_scalar('learning_rate', lr, step)

            # plot results
            bbox_pred = bbox_pred.data[0:1].cpu().numpy()
            iou_pred = iou_pred.data[0:1].cpu().numpy()
            prob_pred = prob_pred.data[0:1].cpu().numpy()
            image = im[0]
            bboxes, scores, cls_inds = yolo_utils.postprocess(
                bbox_pred, iou_pred, prob_pred, image.shape, cfg, thresh=0.3, size_index=size_index)
            im2show = yolo_utils.draw_detection(image, bboxes, scores, cls_inds, cfg)
            summary_writer.add_image('predict', im2show, step)

        train_loss = 0
        bbox_loss, iou_loss, cls_loss = 0., 0., 0.
        cnt = 0
        t.clear()
        size_index = randint(0, len(cfg.multi_scale_inp_size) - 1)
        #print("image_size {}".format(cfg.multi_scale_inp_size[size_index]))

    if step > 0 and (step % imdb.batch_per_epoch == 0):
        if imdb.epoch in cfg.lr_decay_epochs:
            lr *= cfg.lr_decay
            optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                        momentum=cfg.momentum,
                                        weight_decay=cfg.weight_decay)
        if use_trained_model == 0:
            save_name = os.path.join(cfg.train_output_dir,
                                 '{}_{}.h5'.format(cfg.exp_name, imdb.epoch))
            net_utils.save_net(save_name, net)
            #print(net)

            save_name = os.path.join(cfg.train_output_dir,'{}_{}.pth'.format(cfg.exp_name,imdb.epoch))
    
            torch.save({
                'state': net.state_dict(),
                'epoch': imdb.epoch,
                'lr':lr,
                'optim': optimizer.state_dict(),
            }, save_name)
            print(('save model: {}'.format(save_name)))
        
           
        if best_weights == 1:
            save_name = os.path.join(cfg.train_output_dir,'best_weights.h5')
            net_utils.save_net(save_name, net)
            save_name = os.path.join(cfg.train_output_dir,'best_weights.pth')
            torch.save({
            'state': net.state_dict(),
            'epoch': imdb.epoch,
            'lr':lr,
            'optim': optimizer.state_dict(),
            }, save_name)
            best_weights = 0

        step_cnt = 0

imdb.close()
