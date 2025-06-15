import numpy as np
import pandas as pd
import h5py
import torch
from kmeans_pytorch import kmeans, kmeans_predict
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_path',
                        type=str,
                        default='/public/home/qiuyl/FIDTM-master/save_file/Q/S/Feature256/QtrainQ.h5',
                        help="""
                        Directory to read features
                        """)
    '''
    parser.add_argument('--cluster_num',
                        type=int,
                        default=2 ,
                        help="""
                        Directory to cache
                        """)
    '''

    parser.add_argument('--elbow_map',
                        type=str,
                        default='/public/home/qiuyl/FIDTM-master/image/Q_elbow_map',
                        help="""
                        Directory to save cluster result
                        """)                   


    args = parser.parse_args()
    return args
if __name__ == '__main__':
    # 导入数据
    args = parse_args()
    #num_clusters = args.cluster_num
    feature_path = args.feature_path
    h5f = h5py.File(feature_path,'r') # open this file.h5
    feature = h5f['features'] # get features
    feature = np.array(feature)
    data = np.reshape(feature,(len(feature),224,224,1))
    data =np.squeeze(data)
    # 设定：数据集数量，数据集维数，聚类的类别数
    # data_size, dims = len(data), 3
    print('data_shape:',data.shape)
    data = torch.from_numpy(data)
    reshaped_data = torch.reshape(data, (len(feature), 224*224))

    # print('data_shape:',data.shape())

    # 训练阶段
    # X：待聚类数据集(需要是torch.Tensor类型)，维数，距离计算法则，训练设备
    distortions = []  # 用来存放设置不同簇数时的SSE值
    for i in range(2,11):
        kmModel = KMeans(n_clusters=i)
        kmModel.fit(reshaped_data)
        distortions.append(kmModel.inertia_)  # 获取K-means算法的SSE
    # 绘制曲线
    plt.plot(range(2, 11), distortions, marker="o")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel("簇数量")
    plt.ylabel("簇内误方差(SSE)")
    plt.savefig(args.elbow_map)

    # # ======================================================================================================================
    # # plot：绘图阶段————训练集上的聚类图
    # # 使用 t-SNE 进行降维
    # tsne = TSNE(n_components=2, random_state=42)
    # data_tsne = tsne.fit_transform(reshaped_data)
    # center_tsne = tsne.fit_transform(cluster_centers)
    # # 绘制散点图，并根据聚类结果着色
    # plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=cluster_ids_x, cmap='spring')

    # # 绘制聚类中心
    # # center_tsne = tsne.fit_transform(cluster_centers)
    # plt.scatter(center_tsne[:, 0], center_tsne[:, 1], marker='x', color='red', label='Cluster Centers')

    # plt.legend()
    # # plt.show()
    # # 聚类中心点的分布
    # # plt.scatter(
    # #     cluster_centers[:, 0], cluster_centers[:, 1],
    # #     c='white',
    # #     alpha=0.6,
    # #     edgecolors='black',
    # #     linewidths=2
    # # )

    # # plt.tight_layout()
    # # plt.show()
    # plt.savefig(args.cluster_map)

