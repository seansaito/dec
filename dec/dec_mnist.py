import os
os.environ['PATH'] = '../caffe/build/tools:'+os.environ['PATH']
import sys
sys.path = ['../caffe/python'] + sys.path

import cv2
import cv
import numpy as np
import shutil
import random
import leveldb
import caffe
from google import protobuf
from caffe.proto import caffe_pb2
from xml.dom import minidom
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import cPickle
import time

class TMM(object):
    """
    TMM class for calculating the p distributions
    and finding the cluster centroids
    """
    def __init__(self, n_components=1, alpha=1):
        self.n_components = n_components
        self.tol = 1e-5
        self.alpha = float(alpha)

    def fit(self, X):
        from sklearn.cluster import KMeans
        kmeans = KMeans(self.n_components, n_init=20)
        kmeans.fit(X)
        self.cluster_centers_ = kmeans.cluster_centers_
        self.covars_ = np.ones(self.cluster_centers_.shape)

    def transform(self, X):
        p = 1.0
        dist = cdist(X, self.cluster_centers_)
        r = 1.0/(1.0+dist**2/self.alpha)**((self.alpha+p)/2.0)
        r = (r.T/r.sum(axis=1)).T
        return r

    def predict(self, X):
        return self.transform(X).argmax(axis=1)

def cluster_acc(Y_pred, Y):
    """
    Finds the cluster accuracy
    """
    from sklearn.utils.linear_assignment_ import linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in xrange(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w

def DisKmeans(db, update_interval = None):
    """
    Training pipeline after autoencoding
    """
    from sklearn.cluster import KMeans
    from sklearn.mixture import GMM
    from sklearn.lda import LDA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import normalized_mutual_info_score
    from scipy.spatial.distance import cdist
    import cPickle
    from scipy.io import loadmat

    if db == 'mnist':
        N_class = 10
        batch_size = 100
        train_batch_size = 256
        X, Y = read_db(db+'_total', True)
        print "==========="
        print Y
        print "==========="
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(np.squeeze(Y), dtype = np.int32)
        N = X.shape[0]
        img = np.clip((X/0.02), 0, 255).astype(np.uint8).reshape((N, 28, 28, 1))

    tmm_alpha = 1.0
    total_iters = (N-1)/train_batch_size+1
    if not update_interval:
        update_interval = total_iters
    Y_pred = np.zeros((Y.shape[0]))
    iters = 0
    seek = 0
    dim = 10

    acc_list = []

    while True:
        print "Cluster optimization iteration", iters
        write_net(db, dim, N_class, "'{:08}'".format(0))
        if iters == 0:
            write_db(np.zeros((N,N_class)), np.zeros((N,)), 'train_weight')
            ret, net = extract_feature('net.prototxt', 'exp/'+db+'/save_iter_100000.caffemodel', ['output'], N, True, 0)
            feature = ret[0].squeeze()

            gmm_model = TMM(N_class)
            gmm_model.fit(feature)
            net.params['loss'][0].data[0,0,:,:] = gmm_model.cluster_centers_.T
            net.params['loss'][1].data[0,0,:,:] = 1.0/gmm_model.covars_.T
        else:
            ret, net = extract_feature('net.prototxt', 'init.caffemodel', ['output'], N, True, 0)
            feature = ret[0].squeeze()

            gmm_model.cluster_centers_ = net.params['loss'][0].data[0,0,:,:].T


        Y_pred_last = Y_pred
        Y_pred = gmm_model.predict(feature).squeeze()
        acc, freq = cluster_acc(Y_pred, Y)
        acc_list.append(acc)
        nmi = normalized_mutual_info_score(Y, Y_pred)
        print freq
        print freq.sum(axis=1)
        print 'acc: ', acc, 'nmi: ', nmi
        print (Y_pred != Y_pred_last).sum()*1.0/N
        if (Y_pred != Y_pred_last).sum() < 0.001*N:
            print acc_list
            return acc, nmi
        time.sleep(1)

        write_net(db, dim, N_class, "'{:08}'".format(seek))
        weight = gmm_model.transform(feature)

        weight = (weight.T/weight.sum(axis=1)).T
        bias = (1.0/weight.sum(axis=0))
        bias = N_class*bias/bias.sum()
        weight = (weight**2)*bias
        weight = (weight.T/weight.sum(axis=1)).T
        print weight[:10,:]
        write_db(weight, np.zeros((weight.shape[0],)), 'train_weight')

        net.save('init.caffemodel')
        del net

        with open('solver.prototxt', 'w') as fsolver:
            fsolver.write("""net: "net.prototxt"
                base_lr: 0.01
                lr_policy: "step"
                gamma: 0.1
                stepsize: 100000
                display: 10
                max_iter: %d
                momentum: 0.9
                weight_decay: 0.0000
                snapshot: 100
                snapshot_prefix: "exp/test/save"
                snapshot_after_train:true
                solver_mode: GPU
                debug_info: false
                sample_print: false
                device_id: 0"""%update_interval)
        os.system('caffe train --solver=solver.prototxt --weights=init.caffemodel')
        shutil.copyfile('exp/test/save_iter_%d.caffemodel'%update_interval, 'init.caffemodel')

        iters += 1
        seek = (seek + train_batch_size*update_interval)%N

"""
DB functions
"""
def read_db(str_db, float_data = True):
    db = leveldb.LevelDB(str_db)
    datum = caffe_pb2.Datum()
    array = []
    label = []
    for k,v in db.RangeIter():
        dt = datum.FromString(v)
        if float_data:
          array.append(dt.float_data)
        else:
          array.append(np.fromstring(dt.data, dtype=np.uint8))
        label.append(dt.label)
    return np.asarray(array), np.asarray(label)

def write_db(X, Y, fname):
    if os.path.exists(fname):
      shutil.rmtree(fname)
    assert X.shape[0] == Y.shape[0]
    X = X.reshape((X.shape[0], X.size/X.shape[0], 1, 1))
    db = leveldb.LevelDB(fname)

    for i in xrange(X.shape[0]):
      x = X[i]
      if x.ndim != 3:
        x = x.reshape((x.size,1,1))
      db.Put('{:08}'.format(i), caffe.io.array_to_datum(x, int(Y[i])).SerializeToString())
    del db

def update_db(seek, N, X, Y, fname):
    assert X.shape[0] == Y.shape[0]
    X = X.reshape((X.shape[0], X.size/X.shape[0], 1, 1))
    db = leveldb.LevelDB(fname)

    for i in xrange(X.shape[0]):
      x = X[i]
      if x.ndim != 3:
        x = x.reshape((x.size,1,1))
      db.Put('{:08}'.format((i+seek)%N), caffe.io.array_to_datum(x, int(Y[i])).SerializeToString())
    del db

"""
Caffe network functions
"""
def extract_feature(net, model, blobs, N, train = False, device = None):
    if type(net) is str:
        if train:
            caffe.Net.set_phase_train()
        if model:
            net = caffe.Net(net, model)
        else:
            net = caffe.Net(net)
        caffe.Net.set_phase_test()
    if not (device is None):
        caffe.Net.set_mode_gpu()
        caffe.Net.set_device(device)

    batch_size = net.blobs[blobs[0]].num
    res = [ [] for i in blobs ]
    for i in xrange((N-1)/batch_size+1):
        ret = net.forward(blobs=blobs)
        for i in xrange(len(blobs)):
            res[i].append(ret[blobs[i]].copy())

    for i in xrange(len(blobs)):
        res[i] = np.concatenate(res[i], axis=0)[:N]

    return res, net

def write_net(db, dim, n_class, seek):
    layers = [ ('data_seek', ('data','dummy',db+'_total', db+'_total', 1.0, seek)),
             ('data_seek', ('label', 'dummy', 'train_weight', 'train_weight', 1.0, seek)),

             ('inner', ('inner1', 'data', 500)),
             ('relu', ('inner1',)),

             ('inner', ('inner2', 'inner1', 500)),
             ('relu', ('inner2',)),

             ('inner', ('inner3', 'inner2', 2000)),
             ('relu', ('inner3',)),

             ('inner', ('output', 'inner3', dim)),

             ('tloss', ('loss', 'output', 'label', n_class))
          ]
    with open('net.prototxt', 'w') as fnet:
        make_net(fnet, layers)


def make_net(fnet, layers):
    layer_dict = {}
    layer_dict['data'] = """layers {{
        name: "{0}"
        type: DATA
        top: "{0}"
        data_param {{
            source: "{2}"
            backend: LEVELDB
            batch_size: 256
        }}
        transform_param {{
            scale: {4}
        }}
        include: {{ phase: TRAIN }}
    }}
    layers {{
        name: "{0}"
        type: DATA
        top: "{0}"
        data_param {{
            source: "{3}"
            backend: LEVELDB
            batch_size: 100
        }}
        transform_param {{
            scale: {4}
        }}
        include: {{ phase: TEST }}
    }}
    """
    layer_dict['data_seek'] = """layers {{
        name: "{0}"
        type: DATA
        top: "{0}"
        data_param {{
            seek: {5}
            source: "{2}"
            backend: LEVELDB
            batch_size: 256
        }}
        transform_param {{
            scale: {4}
        }}
        include: {{ phase: TRAIN }}
    }}
    layers {{
        name: "{0}"
        type: DATA
        top: "{0}"
        data_param {{
            seek: {5}
            source: "{3}"
            backend: LEVELDB
            batch_size: 100
        }}
        transform_param {{
            scale: {4}
        }}
        include: {{ phase: TEST }}
    }}
    """
    layer_dict['sil'] = """layers {{
      name: "{0}silence"
      type: SILENCE
      bottom: "{0}"
    }}
    """
    layer_dict['tloss'] = """layers {{
      name: "{0}"
      type: MULTI_T_LOSS
      bottom: "{1}"
      bottom: "{2}"
      blobs_lr: 1.
      blobs_lr: 0.
      blobs_lr: 0.
      top: "loss"
      top: "std"
      top: "ind"
      top: "proba"
      multi_t_loss_param {{
        num_center: {3}
        alpha: 1
        lambda: 2
        beta: 1
        bandwidth: 0.1
        weight_filler {{
          type: 'gaussian'
          std: 0.5
        }}
      }}
    }}
    layers {{
      name: "silence"
      type: SILENCE
      bottom: "label"
      bottom: "ind"
      bottom: "proba"
    }}
    """
    layer_dict['inner'] = """layers {{
      name: "{0}"
      type: INNER_PRODUCT
      bottom: "{1}"
      top: "{0}"
      blobs_lr: 1
      blobs_lr: 2
      weight_decay: 1
      weight_decay: 0
      inner_product_param {{
        num_output: {2}
        weight_filler {{
          type: "gaussian"
          std: 0.05
        }}
        bias_filler {{
          type: "constant"
          value: 0
        }}
      }}
    }}
    """
    layer_dict['inner_init'] = """layers {{
      name: "{0}"
      type: INNER_PRODUCT
      bottom: "{1}"
      top: "{0}"
      blobs_lr: 1
      blobs_lr: 2
      weight_decay: 1
      weight_decay: 0
      inner_product_param {{
        num_output: {2}
        weight_filler {{
          type: "gaussian"
          std: {3}
        }}
        bias_filler {{
          type: "constant"
          value: 0
        }}
      }}
    }}
    """
    layer_dict['inner_lr'] = """layers {{
      name: "{0}"
      type: INNER_PRODUCT
      bottom: "{1}"
      top: "{0}"
      blobs_lr: {4}
      blobs_lr: {5}
      weight_decay: 1
      weight_decay: 0
      inner_product_param {{
        num_output: {2}
        weight_filler {{
          type: "gaussian"
          std: {3}
        }}
        bias_filler {{
          type: "constant"
          value: 0
        }}
      }}
    }}
    """
    layer_dict['relu'] = """layers {{
      name: "{0}relu"
      type: RELU
      bottom: "{0}"
      top: "{0}"
    }}
    """
    layer_dict['drop'] = """layers {{
      name: "{0}drop"
      type: DROPOUT
      bottom: "{0}"
      top: "{0}"
      dropout_param {{
        dropout_ratio: {1}
      }}
    }}
    """
    layer_dict['drop_copy'] = """layers {{
      name: "{0}drop"
      type: DROPOUT
      bottom: "{1}"
      top: "{0}"
      dropout_param {{
        dropout_ratio: {2}
      }}
    }}
    """
    layer_dict['euclid'] = """layers {{
      name: "{0}"
      type: EUCLIDEAN_LOSS
      bottom: "{1}"
      bottom: "{2}"
      top: "{0}"
    }}
    """

    fnet.write('name: "net"\n')
    for k,v in layers:
        fnet.write(layer_dict[k].format(*v))
    fnet.close()

if __name__ == "__main__":
    lam = 160
    db = "mnist"
    DisKmeans(db, lam)
