exp_name = 'darknet19_trainval_224'

#pretrained_fname = 'yolov2-tiny-224.h5' #darknet19.weights.npz'

start_step = 0
lr_decay_epochs = {60, 90}
lr_decay = 1./10

max_epoch = 13000 #20000넘어가면 메모리부족

weight_decay = 0.0005
momentum = 0.9
init_learning_rate = 1e-3

# for training yolo2
object_scale = 5.
noobject_scale = 1.
class_scale = 1.
coord_scale = 1.
iou_thresh = 0.6

# dataset
imdb_train = 'ryu_2021_trainval' #'voc_2007_trainval'
imdb_test = 'ryu_2021_test' #'voc_2007_test'
batch_size = 1
train_batch_size = 16
