import numpy as np
import pandas as pd
import h5py
import torch
from kmeans_pytorch import kmeans, kmeans_predict
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse
import oot2 as ot
import os
import time
import gc

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query',
                        type=str,
                        default='/HOME/scw6cs4/run/FIDT_main/save_file/B/T/Feature256/AtrainB.h5',
                        help="""
                        Directory to read images
                        """)
    parser.add_argument('--gallery',
                        type=str,
                        default='/HOME/scw6cs4/run/FIDT_main/save_file/A/S/Feature256/AtrainA.h5',
                        help="""
                        Directory to cache
                        """)
    parser.add_argument('--index_cluster_s',
                        type=str,
                        default='/HOME/scw6cs4/run/FIDT_main/cluster_result/A_S_A2B_4.npy',
                        help="""
                        an h5 file to record cluster results of source  category index
                        """)
    parser.add_argument('--center_cluster_s',
                        type=str,
                        default='/HOME/scw6cs4/run/FIDT_main/cluster_result/A_S_A2B_4.h5',
                        help="""
                        an h5 file to record cluster results of source  cluster center
                        """)
    parser.add_argument('--index_cluster_t',
                        type=str,
                        default='/HOME/scw6cs4/run/FIDT_main/cluster_result/B_T_A2B_256_4.npy',
                        help="""
                        an h5 file to record cluster results of target including category and cluster center
                        """)
    parser.add_argument('--center_cluster_t',
                        type=str,
                        default='/HOME/scw6cs4/run/FIDT_main/cluster_result/B_T_A2B_256_4.h5',
                        help="""
                        an h5 file to record cluster results of target including category and cluster center
                        """)
    parser.add_argument('--regularization_parameter1',
                        type=float,
                        default=1e-3,
                        help='set regularization accuracy for OT')
    parser.add_argument('--regularization_parameter2',
                        type=float,
                        default=1e-3,
                        help='set regularization accuracy for OT')
    # parser.add_argument('--index',
    #                     type=str,
    #                     default='/HOME/scw6cs4/run/FIDT_main/save_file/out/A2B_sparse_columnsum.h5',
    #                     help="""
    #                         Directory to alpha
    #                         """)
    parser.add_argument('--selected_result',
                        type=str,
                        default='/HOME/scw6cs4/run/FIDT_main/npydata/A2B_cluster_ot.npy',
                        help="""
                        an npy to record selected files of source domain
                        """)
    parser.add_argument('--corresponding_weight',
                        type=str,
                        default='/HOME/scw6cs4/run/FIDT_main/npydata/A2B_cluster_ot_weight.npy' ,
                        help="""
                        an npy to record selected files of source domain
                        """)
    args = parser.parse_args()

    return args


def record_file_name(args,mapping,weight):
    h5f = h5py.File(args.gallery,'r')
    gallery_path = h5f['paths'][:] # get paths
    train_list = []
    weight_list = []
    for i in range(len(mapping)):
        images_path = str(gallery_path[mapping[i]])[2:-1]
        # print('image_path:',images_path)
        if(images_path not in train_list):
            train_list.append(images_path)
            weight_list.append(weight[i])
            print('image_path:',images_path)
            print('weight:', weight[i])
    # train_list.sort()
    print('train_list_len:',len(train_list))

    return train_list,weight_list





if __name__ == '__main__':
    # 导入数据
    args = parse_args()
    #输出所有参数
    for arg, value in args.__dict__.items():
        print(f"{arg}: {value}")
    Train_list=[] #记录选中的源域文件
    Weight_list=[]
    h5f = h5py.File(args.center_cluster_s,'r') # open this file.h5
    center_s = h5f['cluster_center'] # get features
    center_s = np.array(center_s)
    Y=np.reshape(center_s,(len(center_s),224,224,1))
    cluster_s = np.load(args.index_cluster_s, allow_pickle=True)

    f = h5py.File(args.center_cluster_t,'r') # open this file.h5
    center_t = f['cluster_center'] # get features
    center_t = np.array(center_t)
    print('center_t_shape', center_t.shape)
    X=np.reshape(center_t,(len(center_t),224,224,1))
    print('X_shape',X.shape)
    cluster_t = np.load(args.index_cluster_t, allow_pickle=True)
    print('cluster_t_len',len(cluster_t))
    print('cluster_t_type:', type(cluster_t))

    #通过OT得到类与类之间的mapping
   
    numitermanx=2000

    # #这里计算a,b得到不均衡的类比例,a是目标域，b是源域
    # a_list=[]
    # b_list=[]
    # for item in cluster_t:
    #     a_list.append(len(item[0]))
    # total_sum = sum(a_list) # 归一化
    # a_list = [x / total_sum for x in a_list]
    # a_array = np.array(a_list)
    # for item in cluster_s:
    #     b_list.append(len(item[0]))
    # total_sum = sum(b_list) # 归一化
    # b_list = [x / total_sum for x in b_list]
    # b_array = np.array(b_list)
    begin = time.time()
    print('------begin_OT_time-------',begin,flush=True)

    mapping = ot.optimal_transport_inter_2(X,Y,args.regularization_parameter1,numintermax=numitermanx)
    #读取特征
    h5f = h5py.File(args.gallery, 'r')
    s_feature = h5f['features'][:]
    s_feature = np.array(s_feature)
    s_feature = np.reshape(s_feature, (len(s_feature), 224, 224, 1))
    f = h5py.File(args.query, 'r')
    t_feature = f['features'][:]
    t_feature = np.array(t_feature)
    t_feature = np.reshape(t_feature, (len(t_feature), 224, 224, 1))
    print('t_feature.shape:',t_feature.shape)
    #类间OT
    cluster_num=len(X)
    save_mapping=[]
    for i in range(len(X)):#遍历每个类
       index=mapping[i]
       print('index:',index)
       #根据索引序列取出对应的特征
       # print('type(cluster_t[i]):', type(cluster_t[i]))
       print('cluster_index_target')
       #for item in cluster_t[i][0]:
       for item in cluster_t[i]:
           print(item)
       # cluster_t_feature = [t_feature[j] for j in cluster_t[i][0]]
       cluster_t_feature=[t_feature[j] for j in cluster_t[i]]
       cluster_t_feature = np.array(cluster_t_feature)
       print('cluster_t_feature', cluster_t_feature.shape)
       # cluster_t[i] = np.reshape(cluster_t[i], (len(cluster_t[i]), 224, 224, 1))
       # print('cluster_t_feature_shape:',cluster_t[i].shape)
       #cluster_s_feature = [s_feature[j] for j in cluster_s[index][0]]
       cluster_s_feature = [s_feature[j] for j in cluster_s[index]]
       cluster_s_feature = np.array(cluster_s_feature)
       # cluster_s[index] = np.reshape(cluster_s[index], (len(cluster_s[index]), 224, 224, 1))
       print('cluster_s_feature_shape:', cluster_s_feature.shape)
       mapping_intra,weight = ot.optimal_transport_intra_2(cluster_t_feature,cluster_s_feature,args.regularization_parameter2)
       print('len(mapping)',len(mapping))
       #得到选中的文件在源域中的索引
       mapping_final=[cluster_s[index][j] for j in mapping_intra]
       save_mapping.extend(mapping_final)
       print('len(mapping_final)',len(mapping_final))
       print('mapping_index_source')
       for item in mapping_final:
           print(item)
       #得到某类的源域文件列表
       category_file,weight_file= record_file_name(args,mapping_final,weight)
       Train_list.extend(category_file)
       Weight_list.extend(weight_file)

       #这里用到了内存回收机制
       del cluster_s_feature, cluster_t_feature, mapping_intra, weight, mapping_final
       gc.collect()

    end = time.time()
    print('------end_OT_time-------',end, flush=True)

    # index = args.index
    # dirs = os.path.dirname(index)
    # if not os.path.exists(dirs): 
    #     os.makedirs(dirs)
    # h = h5py.File(index,'a')
    # h.create_dataset('index',data=save_mapping)
    # h.close()

    #将文件列表保存为npy文件
    np.save(args.selected_result, Train_list)
    np.save(args.corresponding_weight, Weight_list)










