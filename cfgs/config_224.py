import numpy as np

#name = 'models/training/darknet_19_trainval_224/darknet19_trainavl_224_2.h5'
name = 'models/training/best_weights.h5'
# trained model
h5_fname = name
# myflag
label_names = ('0',)
num_classes = len(label_names)
best_weights = 100
#anchors = np.asarray([(0.57273, 0.677385), (1.87446, 2.06253),
#                      (3.33843, 5.47434), (6.88282, 3.52778), (6.77052, 6.16828)],
#                      dtype=np.float)

#anchors = np.asarray([(0.2245,0.9709), (0.3463,1.7239), (0.5014,2.6883),
#                         (0.7388,4.0743), (1.1526,4.7111)],dtype=np.float)

#anchors = np.asarray([(0.57273, 0.677385), (1.87446, 2.06253),
#                      (3.33843, 5.47434), (0.7388,4.0743), (1.1526,4.7111)],
#                      dtype=np.float)

#anchors = np.asarray([(0.57273, 0.677385), (1.87446, 2.06253),
#                      (3.33843, 5.47434), (3.7388,1.0743), (3.1526,3.7111)],
#                      dtype=np.float)

#anchors = np.asarray([(0.57273, 0.677385), (1.87446, 2.06253),
#                      (3.33843, 5.47434), (0.3463,0.6237), (2.4526,3.1111)],
#                      dtype=np.float)

anchors = np.asarray([(0.57273, 0.677385), (1.87446, 2.06253),
                      (3.33843, 5.47434), (0.3463,0.6237), (1.4526,3.1111)],
                      dtype=np.float)


num_anchors = len(anchors)
