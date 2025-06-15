import numpy as np
import pandas as pd
import h5py
import torch
from kmeans_pytorch import kmeans, kmeans_predict
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import argparse
import oot2 as ot
from sklearn.decomposition import PCA
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature',
                        type=str,
                        default='/HOME/scw6cs4/run/FIDT_main/save_file/A2B_sparse.h5',
                        help="""
                        Directory to read images
                        """)
    parser.add_argument('--cluster_num',
                        type=int,
                        default=4,
                        help="""
                        Directory to read images
                        """)
    parser.add_argument('--center_cluster',
                        type=str,
                        default='/HOME/scw6cs4/run/FIDT_main/cluster_result/A_S_A2B_sparse_4.h5',
                        help="""
                        an h5 file to record cluster results of cluster center
                        """)
    parser.add_argument('--cluster_index',
                        type=str,
                        default='/HOME/scw6cs4/run/FIDT_main/cluster_result/A_S_A2B_sparse_4.npy',
                        help="""
                        an h5 file to record cluster results of category index
                        """)
    parser.add_argument('--vis_path',
                        type=str,
                        default='/public/home/qiuyl/FIDTM-master/image/Q2B_B_cluster_tsne.png',
                        help="""
                        an png file to record cluster visualization 
                        """)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    num_clusters = args.cluster_num
    h5f = h5py.File(args.feature, 'r')
    feature = h5f['features'][:]
    paths = h5f['paths'][:]
    print(feature.shape)
    reshaped_data_numpy = feature.reshape(len(feature),50176)
    reshaped_data = torch.from_numpy(reshaped_data_numpy)

    # cluster_ids_x, cluster_centers = kmeans(
    #     X=reshaped_data, num_clusters=num_clusters, distance='euclidean', device=torch.device("cuda:0")
    # )
    cluster_ids_x, cluster_centers = kmeans(
        X=reshaped_data, num_clusters=num_clusters, distance='euclidean'
    )

    #可视化
    # colors = list(mcolors.TABLEAU_COLORS.keys())
    pca = PCA(n_components=2)  # 加载PCA算法，设置降维后主成分数目为2
    data_embedded = pca.fit_transform(reshaped_data)  # 对样本进行降维
    # data_embedded = TSNE(n_components=2).fit_transform(reshaped_data)
    center_embedded = pca.fit_transform(cluster_centers)
    # plt.scatter(data_embedded[:, 0], data_embedded[:, 1], c=cluster_ids_x, cmap='cool') 这种降维方法不对
    colors = list(mcolors.TABLEAU_COLORS.keys())
    N = data_embedded.shape[0]
    #可视化样本
    for i in range(N):
        label = cluster_ids_x[i]
        plt.scatter(data_embedded[i][0], data_embedded[i][1], color=mcolors.TABLEAU_COLORS[colors[label]], s=1)
    #可视化聚类中心
    #plt.scatter(center_embedded[:, 0], center_embedded[:, 1],c='white',alpha=0.6,edgecolors='black',linewidths=2)
    #plt.savefig(args.vis_path)

    '''
    prototype_list = []
    #可视化聚类中心

    for i in range(num_clusters):
        category_index = np.where(cluster_ids_x == i)[0].tolist()
        print('category_index',category_index)
        category_feature = [reshaped_data_numpy[j] for j in category_index]
        category_feature = np.array(category_feature)
        print('category_shape:', category_feature.shape)
        category_feature = np.array(category_feature).squeeze()
        print('category_shape:', category_feature.shape)
        prototype = np.mean(category_feature, axis=0)
        prototype = np.reshape(prototype, (1, 50176)).tolist()
        prototype_list.append(prototype)
        # print('prototype_shape:',prototype.shape)
    prototype_list = np.array(prototype_list).squeeze()
    print('prototype_shape:', prototype_list.shape)
    prototype_embedded = pca.fit_transform(prototype_list)
    plt.scatter(prototype_embedded[:, 0], prototype_embedded[:, 1], c='white', alpha=0.6, edgecolors='black',
                linewidths=2)
    '''

    #按照聚类结果 分类样本
    cluster_indices = []
    for item in range(num_clusters):
        category_index = np.where(cluster_ids_x == item)[0].tolist()
        print('簇的数量：',len(category_index))
        path_cluster = [paths[i] for i in category_index]
        print('path_cluster:',path_cluster)
        cluster_indices.append(category_index)
    cluster_indices = np.array(cluster_indices, dtype=object)

    #保存分类结果

    index = args.center_cluster  # 保存聚类中心
    dirs = os.path.dirname(index)
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    h = h5py.File(index, 'w')
    h.create_dataset('cluster_center', data=cluster_centers)

    np.save(args.cluster_index, cluster_indices) #保存聚类索引
    h.close()


