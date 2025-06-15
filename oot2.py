import numpy as np
import os
import ot
import h5py
from sklearn.decomposition import PCA
from skimage.metrics import structural_similarity as compare_ssim
import cv2
import torch
import torch.nn.functional as F
import argparse
import time
import scipy.stats
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query',
                        type=str,
                        default='/HOME/scw6cs4/run/FIDT_main/save_file/B/T/Feature256/ATrainB.h5',
                        help="""
                        Directory to read images
                        """)
    parser.add_argument('--gallery',
                        type=str,
                        default='/HOME/scw6cs4/run/FIDT_main/save_file/A/S/Feature256/AtrainA.h5' ,
                        help="""
                        Directory to cache
                        """)
    # parser.add_argument('--cindex',
    #                     type=str,
    #                     default='/HOME/scw6cs4/run/FIDT_main/save_file/outC/A2B_256.h5' ,
    #                     help="""
    #                     Directory to ssim
    #                     """)
    # parser.add_argument('--index',
    #                     type=str,
    #                     default='/HOME/scw6cs4/run/FIDT_main/save_file/out/A2B_DAOT.h5' ,
    #                     help="""
    #                     Directory to mapping
    #                     """)
    parser.add_argument('--selected_result',
                        type=str,
                        default='/HOME/scw6cs4/run/FIDT_main/npydata/A2B_ot.npy',
                        help="""
                        an npy to record selected files of source domain
                        """)
    parser.add_argument('--inSize',
                        type=int,
                        default=256 ,
                        help="""
                        """)
    args = parser.parse_args()
    return args
# #欧氏距离
# def euclidean_distance(img1,img2):
#     dist = np.sqrt(np.sum((img1-img2)**2))
#     return dist
# #余弦相似度
# def cosine(img1,img2):
#     dist = np.dot(img1,img2)/(np.sqrt(img1,img1)*np.dot(img2,img2)) 
#     return dist
# #KL散度
# def KL_distance(img1,img2):#但是可能需要先对img1和img2归一化
#     return scipy.stats.entropy(img1,img2)

# #JS散度
# def JS_distance(img1,img2):
#     P=img1
#     Q=img2
#     M=(P+Q)/2
#     return 0.5*scipy.stats.entropy(P, M)+0.5*scipy.stats.entropy(Q, M)


    
#计算目标域patches(一张)和源域patches(一批)的SSIM结构相似性
def ssim(img1, img2s, window_size=256, size_average=True,batch=True):
    img1 = img1.cuda()
    img2s = img2s.cuda()

    mu1 = F.avg_pool2d(img1, window_size, 1, window_size//2, True)
    mu2 = F.avg_pool2d(img2s, window_size, 1, window_size//2, True)
    sigma1 = F.avg_pool2d(img1 ** 2, window_size, 1, window_size//2, True) - mu1 ** 2
    sigma2 = F.avg_pool2d(img2s ** 2, window_size, 1, window_size//2, True) - mu2 ** 2
    sigma12 = F.avg_pool2d(img1 * img2s, window_size, 1, window_size//2, True) - mu1 * mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
    # print('----------ssim_map_shape:----------',ssim_map.shape)
    if size_average:
        if batch:
            return 1- ssim_map.mean(dim=(1, 2, 3))
        else:
            return 1- ssim_map.mean()
    else:
        return 1- ssim_map.mean(1).mean(1).mean(1)

def ssim_2(img1, img2, window_size=256, size_average=True):
    img1 = img1.cuda()
    img2 = img2.cuda()
    mu1 = F.avg_pool2d(img1, window_size, 1, window_size//2, True)
    mu2 = F.avg_pool2d(img2, window_size, 1, window_size//2, True)
    sigma1 = F.avg_pool2d(img1 ** 2, window_size, 1, window_size//2, True) - mu1 ** 2
    sigma2 = F.avg_pool2d(img2 ** 2, window_size, 1, window_size//2, True) - mu2 ** 2
    sigma12 = F.avg_pool2d(img1 * img2, window_size, 1, window_size//2, True) - mu1 * mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
    # print('ssim_type:',type(ssim_map))
    # print('ssim_mean:',type(ssim_map.mean()))
    # print('ssim_mean_t:',ssim_map.mean())
    if size_average:
        # result=1- ssim_map.mean()
        # print(type(result))
        # print(result)
        return 1- ssim_map.mean()
    else:
        return 1- ssim_map.mean(1).mean(1).mean(1)




def optimal_transport(X,Y,regulation,index,numintermax=1000):
    
    # gallery = np.array(gallery)
    # query = np.array(query)
    # Y = np.reshape(gallery,(len(gallery),224,224,1)) #源域
    # X = np.reshape(query,(len(query),224,224,1)) #目标域
    C = np.zeros((len(X), len(Y)))
    print('------------------------------------------\n'
          '          get_loss_matrix                 \n'
          '------------------------------------------',flush=True)
    # start = time.time()
    # print('------start_time-------',start)
    # C[0:len(X), len(Y)]=ssim(torch.tensor(X), torch.tensor(np.ones((224, 224, 1))))

    batch_size=10000 #受限于显存
    try:
        if(len(Y)<batch_size): #源域的patches数量较小，不会发生溢出
            for i in range(len(X)):
                C[i,0:len(Y)] = ssim(torch.tensor(X[i]), torch.tensor(Y)).cpu().numpy()
            # end = time.time()
            # print('------end_time-------',end,flush=True)
        else: #源域的patches数量较大，会发生溢出需批量处理
            batch_gallery=len(Y)//batch_size
            for i in range(len(X)):
                C[i, len(Y)]=ssim(torch.tensor(X[i]), torch.tensor(np.ones((224, 224, 1))),batch=False)
                threshold_X = C[i, len(Y)]
                for j in range(batch_gallery):
                    start_inx=j*batch_size
                    end_inx=(j+1)*batch_size
                    C[i,start_inx:end_inx] = ssim(torch.tensor(X[i]), torch.tensor(Y[start_inx:end_inx])).cpu().numpy()
                if len(Y)%batch_size!=0:
                    start_inx=batch_gallery*batch_size
                    C[i,start_inx:-1]=ssim(torch.tensor(X[i]), torch.tensor(Y[start_inx:])).cpu().numpy()
                if np.min(C[i,:-1])>threshold_X:
                    C[i,:-1] = 0
                print(C[i,0])
            # C[len(X), 0:len(Y)]=ssim(torch.tensor(np.ones((224,224,1))), torch.tensor(Y))
            for j in range(batch_gallery):
                start_inx=j*batch_size
                end_inx=(j+1)*batch_size
                C[len(X),start_inx:end_inx] = ssim(torch.tensor(np.ones((224,224,1))), torch.tensor(Y[start_inx:end_inx])).cpu().numpy()
            if len(Y)%batch_size!=0:
                start_inx=batch_gallery*batch_size
                C[len(X),start_inx:-1]=ssim(torch.tensor(np.ones((224,224,1))), torch.tensor(Y[start_inx:])).cpu().numpy()
            for j in range(len(Y)):
                threshold_Y = C[len(X), j]
                if np.min(C[:-1,j])>threshold_Y:
                    C[:-1,j] = 0
            C[len(X), len(Y)] = ssim(torch.tensor(np.ones((224,224,1))),torch.tensor(np.zeros((224,224,1))),batch=False)   
            # end = time.time()
            # print('------end_time-------',end,flush=True)
        # dirs = os.path.dirname(index)
        # if not os.path.exists(dirs): 
        #     os.makedirs(dirs)
        # h = h5py.File(index,'a')
        # h.create_dataset('index',data=C)
        # h.close()
    except RuntimeError as exception:
        if "out of memory" in str(exception):
            print('WARNING: out of memory')
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        else:
            raise exception
    # end = time.time()
    # print('------end_time-------',end)
    print('------------------------------------------\n'
          '       exacute_optimal_transport          \n'
          '------------------------------------------')
    # dirs = os.path.dirname(index)
    # if not os.path.exists(dirs):
    #     os.makedirs(dirs)
    # h = h5py.File(index,'a')
    # h.create_dataset('index',data=C)
    # h.close()
    gamma = float(0.1)  # 正则化参数
    epsilon = float(1e-2)  # 收敛精度
    reg = regulation
    a = np.ones((len(X),)) / len(X)  # 目标域的块概率分布
    b = np.ones((len(Y),)) / len(Y)  # 源域的块概率分布
    alpha= ot.bregman.sinkhorn(a, b, C,reg,numItermax=numintermax)  # 这里使用了ot.sinkhorn()函数
    #保存alpha
    # dirs = os.path.dirname(alpha_path)
    # if not os.path.exists(dirs):
    #     os.makedirs(dirs)
    # h = h5py.File(alpha_path, 'a')
    # h.create_dataset('alpha', data=alpha)
    # h.close()
    if np.max(alpha[:-1,:-1])>0:
        mapping = np.argmax(alpha, axis=1)
    else:
        mapping = np.argmax([],dtype=int)   
    return mapping

def optimal_transport_inter_2(X, Y,regulation, numintermax=1000):
    # gallery = np.array(gallery)
    # query = np.array(query)
    # Y = np.reshape(gallery,(len(gallery),224,224,1)) #源域
    # X = np.reshape(query,(len(query),224,224,1)) #目标域
    C = np.zeros((len(X), len(Y)))
    print('------------------------------------------\n'
          '          get_loss_matrix                 \n'
          '------------------------------------------', flush=True)
    # start = time.time()
    # print('------start_time-------',start)
    # C[0:len(X), len(Y)]=ssim(torch.tensor(X), torch.tensor(np.ones((224, 224, 1))))

    batch_size = 10000  # 受限于显存
    try:
        if (len(Y) < batch_size):  # 源域的patches数量较小，不会发生溢出
            for i in range(len(X)):
                C[i, 0:len(Y)] = ssim(torch.tensor(X[i]), torch.tensor(Y)).cpu().numpy()
            # end = time.time()
            # print('------end_time-------',end,flush=True)
        else:  # 源域的patches数量较大，会发生溢出需批量处理
            batch_gallery = len(Y) // batch_size
            for i in range(len(X)):
                #C[i, len(Y)] = ssim(torch.tensor(X[i]), torch.tensor(np.ones((224, 224, 1))), batch=False)
                #threshold_X = C[i, len(Y)]
                for j in range(batch_gallery):
                    start_inx = j * batch_size
                    end_inx = (j + 1) * batch_size
                    C[i, start_inx:end_inx] = ssim(torch.tensor(X[i]), torch.tensor(Y[start_inx:end_inx])).cpu().numpy()
                if len(Y) % batch_size != 0:
                    start_inx = batch_gallery * batch_size
                    C[i, start_inx:-1] = ssim(torch.tensor(X[i]), torch.tensor(Y[start_inx:])).cpu().numpy()
                #if np.min(C[i, :-1]) > threshold_X:
                    #C[i, :-1] = 0
                #print(C[i, 0])
            # C[len(X), 0:len(Y)]=ssim(torch.tensor(np.ones((224,224,1))), torch.tensor(Y))
            #for j in range(batch_gallery):
                #start_inx = j * batch_size
                #end_inx = (j + 1) * batch_size
                #C[len(X), start_inx:end_inx] = ssim(torch.tensor(np.ones((224, 224, 1))),
                #                                   torch.tensor(Y[start_inx:end_inx])).cpu().numpy()
            #if len(Y) % batch_size != 0:
            #   start_inx = batch_gallery * batch_size
                #C[len(X), start_inx:-1] = ssim(torch.tensor(np.ones((224, 224, 1))),
                #                               torch.tensor(Y[start_inx:])).cpu().numpy()
            # for j in range(len(Y)):
            #     threshold_Y = C[len(X), j]
            #     if np.min(C[:-1, j]) > threshold_Y:
            #         C[:-1, j] = 0
            # C[len(X), len(Y)] = ssim(torch.tensor(np.ones((224, 224, 1))), torch.tensor(np.zeros((224, 224, 1))),
            #                          batch=False)
            # end = time.time()
            # print('------end_time-------',end,flush=True)
        # dirs = os.path.dirname(index)
        # if not os.path.exists(dirs):
        #     os.makedirs(dirs)
        # h = h5py.File(index,'a')
        # h.create_dataset('index',data=C)
        # h.close()
    except RuntimeError as exception:
        if "out of memory" in str(exception):
            print('WARNING: out of memory')
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        else:
            raise exception
    # end = time.time()
    # print('------end_time-------',end)
    print('------------------------------------------\n'
          '       exacute_optimal_transport          \n'
          '------------------------------------------')
    # dirs = os.path.dirname(index)
    # if not os.path.exists(dirs):
    #     os.makedirs(dirs)
    # h = h5py.File(index,'a')
    # h.create_dataset('index',data=C)
    # h.close()
    gamma = float(0.1)  # 正则化参数
    epsilon = float(1e-2)  # 收敛精度
    reg = regulation
    a = np.ones((len(X),)) / len(X)  # 目标域的块概率分布
    b = np.ones((len(Y),)) / len(Y)  # 源域的块概率分布
    alpha = ot.bregman.sinkhorn(a, b, C, reg, numItermax=numintermax)  # 这里使用了ot.sinkhorn()函数
    #alpha = ot.partial.entropic_partial_wasserstein(a, b, C, reg, m=None, stopThr=1e-10, log=False)
    #alpha = ot.unbalanced.sinkhorn_unbalanced(a_array, b_array, C, reg, 1, method='sinkhorn', reg_type='entropy', warmstart=None,numItermax=1000, stopThr=1e-06, verbose=False, log=False)
    #
    print('alpha:',alpha)
    # dirs = os.path.dirname(alpha_path)
    # if not os.path.exists(dirs):
    #     os.makedirs(dirs)
    # h = h5py.File(alpha_path, 'a')
    # h.create_dataset('alpha', data=alpha)
    # h.close()
    if np.max(alpha[:-1, :-1]) > 0:
        mapping = np.argmax(alpha, axis=1)
    else:
        mapping = np.argmax([], dtype=int)
    return mapping

def optimal_transport_inter(X, Y, a_array,b_array,regulation, index, numintermax=1000):
    # gallery = np.array(gallery)
    # query = np.array(query)
    # Y = np.reshape(gallery,(len(gallery),224,224,1)) #源域
    # X = np.reshape(query,(len(query),224,224,1)) #目标域
    C = np.zeros((len(X), len(Y)))
    print('------------------------------------------\n'
          '          get_loss_matrix                 \n'
          '------------------------------------------', flush=True)
    # start = time.time()
    # print('------start_time-------',start)
    # C[0:len(X), len(Y)]=ssim(torch.tensor(X), torch.tensor(np.ones((224, 224, 1))))

    batch_size = 10000  # 受限于显存
    try:
        if (len(Y) < batch_size):  # 源域的patches数量较小，不会发生溢出
            for i in range(len(X)):
                C[i, 0:len(Y)] = ssim(torch.tensor(X[i]), torch.tensor(Y)).cpu().numpy()
            # end = time.time()
            # print('------end_time-------',end,flush=True)
        else:  # 源域的patches数量较大，会发生溢出需批量处理
            batch_gallery = len(Y) // batch_size
            for i in range(len(X)):
                C[i, len(Y)] = ssim(torch.tensor(X[i]), torch.tensor(np.ones((224, 224, 1))), batch=False)
                threshold_X = C[i, len(Y)]
                for j in range(batch_gallery):
                    start_inx = j * batch_size
                    end_inx = (j + 1) * batch_size
                    C[i, start_inx:end_inx] = ssim(torch.tensor(X[i]), torch.tensor(Y[start_inx:end_inx])).cpu().numpy()
                if len(Y) % batch_size != 0:
                    start_inx = batch_gallery * batch_size
                    C[i, start_inx:-1] = ssim(torch.tensor(X[i]), torch.tensor(Y[start_inx:])).cpu().numpy()
                if np.min(C[i, :-1]) > threshold_X:
                    C[i, :-1] = 0
                print(C[i, 0])
            # C[len(X), 0:len(Y)]=ssim(torch.tensor(np.ones((224,224,1))), torch.tensor(Y))
            for j in range(batch_gallery):
                start_inx = j * batch_size
                end_inx = (j + 1) * batch_size
                C[len(X), start_inx:end_inx] = ssim(torch.tensor(np.ones((224, 224, 1))),
                                                    torch.tensor(Y[start_inx:end_inx])).cpu().numpy()
            if len(Y) % batch_size != 0:
                start_inx = batch_gallery * batch_size
                C[len(X), start_inx:-1] = ssim(torch.tensor(np.ones((224, 224, 1))),
                                               torch.tensor(Y[start_inx:])).cpu().numpy()
            for j in range(len(Y)):
                threshold_Y = C[len(X), j]
                if np.min(C[:-1, j]) > threshold_Y:
                    C[:-1, j] = 0
            C[len(X), len(Y)] = ssim(torch.tensor(np.ones((224, 224, 1))), torch.tensor(np.zeros((224, 224, 1))),
                                     batch=False)
            # end = time.time()
            # print('------end_time-------',end,flush=True)
        # dirs = os.path.dirname(index)
        # if not os.path.exists(dirs):
        #     os.makedirs(dirs)
        # h = h5py.File(index,'a')
        # h.create_dataset('index',data=C)
        # h.close()
    except RuntimeError as exception:
        if "out of memory" in str(exception):
            print('WARNING: out of memory')
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        else:
            raise exception
    # end = time.time()
    # print('------end_time-------',end)
    print('------------------------------------------\n'
          '       exacute_optimal_transport          \n'
          '------------------------------------------')
    # dirs = os.path.dirname(index)
    # if not os.path.exists(dirs):
    #     os.makedirs(dirs)
    # h = h5py.File(index,'a')
    # h.create_dataset('index',data=C)
    # h.close()
    gamma = float(0.1)  # 正则化参数
    epsilon = float(1e-2)  # 收敛精度
    reg = regulation
    #a = np.ones((len(X),)) / len(X)  # 目标域的块概率分布
    #b = np.ones((len(Y),)) / len(Y)  # 源域的块概率分布
    #alpha = ot.bregman.sinkhorn(a, b, C, reg, numItermax=numintermax)  # 这里使用了ot.sinkhorn()函数
    #alpha = ot.partial.entropic_partial_wasserstein(a, b, C, reg, m=None, stopThr=1e-10, log=False)
    alpha = ot.unbalanced.sinkhorn_unbalanced(a_array, b_array, C, reg, 1, method='sinkhorn', reg_type='entropy', warmstart=None,numItermax=1000, stopThr=1e-06, verbose=False, log=False)
    #
    print('alpha:',alpha)
    # dirs = os.path.dirname(alpha_path)
    # if not os.path.exists(dirs):
    #     os.makedirs(dirs)
    # h = h5py.File(alpha_path, 'a')
    # h.create_dataset('alpha', data=alpha)
    # h.close()
    if np.max(alpha[:-1, :-1]) > 0:
        mapping = np.argmax(alpha, axis=1)
    else:
        mapping = np.argmax([], dtype=int)
    return mapping


def optimal_transport_intra(X, Y, regulation, index, numintermax=1000):#不仅返回mapping，还返回对应的权重
    # gallery = np.array(gallery)
    # query = np.array(query)
    # Y = np.reshape(gallery,(len(gallery),224,224,1)) #源域
    # X = np.reshape(query,(len(query),224,224,1)) #目标域
    C = np.zeros((len(X), len(Y)))
    print('------------------------------------------\n'
          '          get_loss_matrix                 \n'
          '------------------------------------------', flush=True)
    # start = time.time()
    # print('------start_time-------',start)
    # C[0:len(X), len(Y)]=ssim(torch.tensor(X), torch.tensor(np.ones((224, 224, 1))))

    batch_size = 10000  # 受限于显存
    try:
        if (len(Y) < batch_size):  # 源域的patches数量较小，不会发生溢出
            for i in range(len(X)):
                C[i, 0:len(Y)] = ssim(torch.tensor(X[i]), torch.tensor(Y)).cpu().numpy()
            # end = time.time()
            # print('------end_time-------',end,flush=True)
        else:  # 源域的patches数量较大，会发生溢出需批量处理
            batch_gallery = len(Y) // batch_size
            for i in range(len(X)):
                C[i, len(Y)] = ssim(torch.tensor(X[i]), torch.tensor(np.ones((224, 224, 1))), batch=False)
                threshold_X = C[i, len(Y)]
                for j in range(batch_gallery):
                    start_inx = j * batch_size
                    end_inx = (j + 1) * batch_size
                    C[i, start_inx:end_inx] = ssim(torch.tensor(X[i]), torch.tensor(Y[start_inx:end_inx])).cpu().numpy()
                if len(Y) % batch_size != 0:
                    start_inx = batch_gallery * batch_size
                    C[i, start_inx:-1] = ssim(torch.tensor(X[i]), torch.tensor(Y[start_inx:])).cpu().numpy()
                if np.min(C[i, :-1]) > threshold_X:
                    C[i, :-1] = 0
                print(C[i, 0])
            # C[len(X), 0:len(Y)]=ssim(torch.tensor(np.ones((224,224,1))), torch.tensor(Y))
            for j in range(batch_gallery):
                start_inx = j * batch_size
                end_inx = (j + 1) * batch_size
                C[len(X), start_inx:end_inx] = ssim(torch.tensor(np.ones((224, 224, 1))),
                                                    torch.tensor(Y[start_inx:end_inx])).cpu().numpy()
            if len(Y) % batch_size != 0:
                start_inx = batch_gallery * batch_size
                C[len(X), start_inx:-1] = ssim(torch.tensor(np.ones((224, 224, 1))),
                                               torch.tensor(Y[start_inx:])).cpu().numpy()
            for j in range(len(Y)):
                threshold_Y = C[len(X), j]
                if np.min(C[:-1, j]) > threshold_Y:
                    C[:-1, j] = 0
            C[len(X), len(Y)] = ssim(torch.tensor(np.ones((224, 224, 1))), torch.tensor(np.zeros((224, 224, 1))),
                                     batch=False)
            # end = time.time()
            # print('------end_time-------',end,flush=True)
        # dirs = os.path.dirname(index)
        # if not os.path.exists(dirs):
        #     os.makedirs(dirs)
        # h = h5py.File(index,'a')
        # h.create_dataset('index',data=C)
        # h.close()
    except RuntimeError as exception:
        if "out of memory" in str(exception):
            print('WARNING: out of memory')
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        else:
            raise exception
    # end = time.time()
    # print('------end_time-------',end)
    print('------------------------------------------\n'
          '       exacute_optimal_transport          \n'
          '------------------------------------------')
    # dirs = os.path.dirname(index)
    # if not os.path.exists(dirs):
    #     os.makedirs(dirs)
    # h = h5py.File(index,'a')
    # h.create_dataset('index',data=C)
    # h.close()
    gamma = float(0.1)  # 正则化参数
    epsilon = float(1e-2)  # 收敛精度
    reg = regulation
    a = np.ones((len(X),)) / len(X)  # 目标域的块概率分布
    b = np.ones((len(Y),)) / len(Y)  # 源域的块概率分布
    #alpha = ot.bregman.sinkhorn(a, b, C, reg, numItermax=numintermax)  # 这里使用了ot.sinkhorn()函数
    alpha= ot.partial.entropic_partial_wasserstein(a, b, C, reg, m=None,stopThr=1e-10, log=False)
    # 保存alpha
    # dirs = os.path.dirname(alpha_path)
    # if not os.path.exists(dirs):
    #     os.makedirs(dirs)
    # h = h5py.File(alpha_path, 'a')
    # h.create_dataset('alpha', data=alpha)
    # h.close()
    if np.max(alpha[:-1, :-1]) > 0:
        mapping = np.argmax(alpha, axis=1)
        #alpha_max = np.zeros_like(alpha)

        #for i in range(alpha.shape[0]):
            #alpha_max[i, mapping[i]] = alpha[i, mapping[i]]
        column_sum = np.sum(alpha, axis=0)
        print('len of column_sum',len(column_sum))
        weight = [column_sum[i] for i in mapping] # 针对target每行返回对应的权重
        mean_value = sum(weight) / len(weight)

        # 对列表中的所有元素除以平均值
        weight_average = [x / mean_value if x > mean_value else 1 for x in weight]

    else:
        mapping = np.argmax([], dtype=int)
        
    return mapping,weight_average


def optimal_transport_intra_2(X, Y, regulation, numintermax=1000):  # 不仅返回mapping，还返回对应的权重
    # gallery = np.array(gallery)
    # query = np.array(query)
    # Y = np.reshape(gallery,(len(gallery),224,224,1)) #源域
    # X = np.reshape(query,(len(query),224,224,1)) #目标域
    C = np.zeros((len(X), len(Y)))
    print('------------------------------------------\n'
          '          get_loss_matrix                 \n'
          '------------------------------------------', flush=True)
    # start = time.time()
    # print('------start_time-------',start)
    # C[0:len(X), len(Y)]=ssim(torch.tensor(X), torch.tensor(np.ones((224, 224, 1))))

    batch_size = 2000  # 受限于显存
    try:
        if (len(Y) < batch_size):  # 源域的patches数量较小，不会发生溢出
            for i in range(len(X)):
                C[i, 0:len(Y)] = ssim(torch.tensor(X[i]), torch.tensor(Y)).cpu().numpy()
            # end = time.time()
            # print('------end_time-------',end,flush=True)
        else:  # 源域的patches数量较大，会发生溢出需批量处理
            batch_gallery = len(Y) // batch_size
            for i in range(len(X)):
                # C[i, len(Y)] = ssim(torch.tensor(X[i]), torch.tensor(np.ones((224, 224, 1))), batch=False)
                # threshold_X = C[i, len(Y)]
                for j in range(batch_gallery):
                    start_inx = j * batch_size
                    end_inx = (j + 1) * batch_size
                    C[i, start_inx:end_inx] = ssim(torch.tensor(X[i]), torch.tensor(Y[start_inx:end_inx])).cpu().numpy()
                if len(Y) % batch_size != 0:
                    start_inx = batch_gallery * batch_size
                    C[i, start_inx:] = ssim(torch.tensor(X[i]), torch.tensor(Y[start_inx:])).cpu().numpy()
          
            
        # dirs = os.path.dirname(index)
        # if not os.path.exists(dirs):
        #     os.makedirs(dirs)
        # h = h5py.File(index,'a')
        # h.create_dataset('index',data=C)
        # h.close()
    except RuntimeError as exception:
        if "out of memory" in str(exception):
            print('WARNING: out of memory')
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        else:
            raise exception
    # end = time.time()
    # print('------end_time-------',end)
    print('------------------------------------------\n'
          '       exacute_optimal_transport          \n'
          '------------------------------------------')
    # dirs = os.path.dirname(index)
    # if not os.path.exists(dirs):
    #     os.makedirs(dirs)
    # h = h5py.File(index,'a')
    # h.create_dataset('index',data=C)
    # h.close()
    gamma = float(0.1)  # 正则化参数
    epsilon = float(1e-2)  # 收敛精度
    reg = regulation
    a = np.ones((len(X),)) / len(X)  # 目标域的块概率分布
    b = np.ones((len(Y),)) / len(Y)  # 源域的块概率分布
    # alpha = ot.bregman.sinkhorn(a, b, C, reg, numItermax=numintermax)  # 这里使用了ot.sinkhorn()函数
    alpha = ot.partial.entropic_partial_wasserstein(a, b, C, reg, m=None, stopThr=1e-10, log=False)
    # 保存alpha
    # dirs = os.path.dirname(alpha_path)
    # if not os.path.exists(dirs):
    #     os.makedirs(dirs)
    # h = h5py.File(alpha_path, 'a')
    # h.create_dataset('alpha', data=alpha)
    # h.close()
    if np.max(alpha[:-1, :-1]) > 0:
        mapping = np.argmax(alpha, axis=1)
        #alpha_max = np.zeros_like(alpha)

        #for i in range(alpha.shape[0]):
            #alpha_max[i, mapping[i]] = alpha[i, mapping[i]]
        column_sum = np.sum(alpha, axis=0)
        print('len of column_sum',len(column_sum))
        weight = [column_sum[i] for i in mapping] # 针对target每行返回对应的权重
        mean_value = sum(weight) / len(weight)

        # 对列表中的所有元素除以平均值
        weight_average = [x / mean_value if x > mean_value else 1 for x in weight]

    else:
        mapping = np.argmax([], dtype=int)

    return mapping,weight_average


  
    C = np.zeros((len(X) + 1, len(Y) + 1))
    print('------------------------------------------\n'
          '          get_loss_matrix                 \n'
          '------------------------------------------', flush=True)


    batch_size = 10000  # 受限于显存
    try:
        if (len(Y) < batch_size):  # 源域的patches数量较小，不会发生溢出
            for i in range(len(X)):
                C[i, len(Y)] = ssim(torch.tensor(X[i]), torch.tensor(np.ones((224, 224, 1))), batch=False)
                threshold_X = C[i, len(Y)]
                C[i, 0:len(Y)] = ssim(torch.tensor(X[i]), torch.tensor(Y)).cpu().numpy()
                if np.min(C[i, :-1]) > threshold_X:
                    C[i, :-1] = 0
            print(C[i, 0])
            C[len(X), 0:len(Y)] = ssim(torch.tensor(np.ones((224, 224, 1))), torch.tensor(Y)).cpu().numpy()

            for j in range(len(Y)):
                threshold_Y = C[len(X), j]
                if np.min(C[:-1, j]) > threshold_Y:
                    C[:-1, j] = 0
            C[len(X), len(Y)] = ssim(torch.tensor(np.ones((224, 224, 1))), torch.tensor(np.zeros((224, 224, 1))),
                                     batch=False)
            threshold = C[len(X), len(Y)]
       
        else:  # 源域的patches数量较大，会发生溢出需批量处理
            batch_gallery = len(Y) // batch_size
            for i in range(len(X)):
                C[i, len(Y)] = ssim(torch.tensor(X[i]), torch.tensor(np.ones((224, 224, 1))), batch=False)
                threshold_X = C[i, len(Y)]
                for j in range(batch_gallery):
                    start_inx = j * batch_size
                    end_inx = (j + 1) * batch_size
                    C[i, start_inx:end_inx] = ssim(torch.tensor(X[i]), torch.tensor(Y[start_inx:end_inx])).cpu().numpy()
                if len(Y) % batch_size != 0:
                    start_inx = batch_gallery * batch_size
                    C[i, start_inx:-1] = ssim(torch.tensor(X[i]), torch.tensor(Y[start_inx:])).cpu().numpy()
                if np.min(C[i, :-1]) > threshold_X:
                    C[i, :-1] = 0
                print(C[i, 0])
           
            for j in range(batch_gallery):
                start_inx = j * batch_size
                end_inx = (j + 1) * batch_size
                C[len(X), start_inx:end_inx] = ssim(torch.tensor(np.ones((224, 224, 1))),
                                                    torch.tensor(Y[start_inx:end_inx])).cpu().numpy()
            if len(Y) % batch_size != 0:
                start_inx = batch_gallery * batch_size
                C[len(X), start_inx:-1] = ssim(torch.tensor(np.ones((224, 224, 1))),
                                               torch.tensor(Y[start_inx:])).cpu().numpy()
            for j in range(len(Y)):
                threshold_Y = C[len(X), j]
                if np.min(C[:-1, j]) > threshold_Y:
                    C[:-1, j] = 0
            C[len(X), len(Y)] = ssim(torch.tensor(np.ones((224, 224, 1))), torch.tensor(np.zeros((224, 224, 1))),
                                     batch=False)

    except RuntimeError as exception:
        if "out of memory" in str(exception):
            print('WARNING: out of memory')
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        else:
            raise exception
    # end = time.time()
    # print('------end_time-------',end)
    print('------------------------------------------\n'
          '       exacute_optimal_transport          \n'
          '------------------------------------------')
    # dirs = os.path.dirname(index)
    # if not os.path.exists(dirs):
    #     os.makedirs(dirs)
    # h = h5py.File(index,'a')
    # h.create_dataset('index',data=C)
    # h.close()
    gamma = float(0.1)  # 正则化参数
    epsilon = float(1e-2)  # 收敛精度
    reg = regulation
    a = np.ones((len(X) + 1,)) / len(X) + 1  # 目标域的块概率分布
    b = np.ones((len(Y) + 1,)) / len(Y) + 1  # 源域的块概率分布
    alpha = ot.bregman.sinkhorn(a, b, C, reg)  # 这里使用了ot.sinkhorn()函数
    # 保存alpha
    dirs = os.path.dirname(alpha_path)
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    h = h5py.File(alpha_path, 'a')
    h.create_dataset('alpha', data=alpha)
    h.close()
    if np.max(alpha[:-1, :-1]) > 0:
        mapping = np.argmax(alpha[:-1, :-1], axis=1)
    else:
        mapping = np.argmax([], dtype=int)
    return mapping

def optimal_transport_2(query,gallery,index):
    gallery = np.array(gallery)
    query = np.array(query)
    Y = np.reshape(gallery,(len(gallery),224,224,1)) #源域
    X = np.reshape(query,(len(query),224,224,1)) #目标域
    C = np.zeros((len(X)+1, len(Y)+1))
    print('------------------------------------------\n'
          '          get_loss_matrix                 \n'
          '------------------------------------------')
    start = time.time()
    print('------start_time-------',start)
    # C[0:len(X), len(Y)]=ssim(torch.tensor(X), torch.tensor(np.ones((224, 224, 1))))

    batch_size=8000 #受限于内存
    if(len(gallery)<batch_size): #源域的patches数量较小，不会发生溢出
        for i in range(len(X)):
            C[i, len(Y)]=ssim_2(torch.tensor(X[i]), torch.tensor(np.ones((224, 224, 1))))
            threshold_X = C[i, len(Y)]
            for j in range(len(Y)):
                C[i,j] = ssim_2(torch.tensor(X[i]), torch.tensor(Y[j]))
                # print(i,j)
                # print(torch.tensor(X[i]),torch.tensor(Y[j]))
                print(C[i, j])
            if np.min(C[i,:-1])>threshold_X:
                C[i,:-1] = 0
        for j in range(len(Y)):
            C[len(X), j] = ssim_2(torch.tensor(np.ones((224,224,1))),torch.tensor(Y[j]))#设置每列的阈值
            threshold_Y = C[len(X), j]
            if np.min(C[:-1,j])>threshold_Y:
                C[:-1,j] = 0
        C[len(X), len(Y)] = ssim_2(torch.tensor(np.ones((224,224,1))),torch.tensor(np.zeros((224,224,1))))
        end = time.time()
        print('------end_time-------',end)
    else: #源域的patches数量较大，会发生溢出需批量处理
        batch_gallery=len(gallery)//batch_size
        for i in range(len(X)):
            C[i, len(Y)]=ssim_2(torch.tensor(X[i]), torch.tensor(np.ones((224, 224, 1))),batch=False)
            threshold_X = C[i, len(Y)]
            for j in range(batch_gallery):
                start_inx=j*batch_size
                end_inx=(j+1)*batch_size
                C[i,start_inx:end_inx] = ssim(torch.tensor(X[i]), torch.tensor(Y[start_inx:end_inx]))
            if len(gallery)%batch_size!=0:
                start_inx=batch_gallery*batch_size
                C[i,start_inx:-1]=ssim(torch.tensor(X[i]), torch.tensor(Y[start_inx:]))
            if torch.min(C[i,:-1])>threshold_X:
                C[i,:-1] = 0
        # C[len(X), 0:len(Y)]=ssim(torch.tensor(np.ones((224,224,1))), torch.tensor(Y))
        for j in range(batch_gallery):
            start_inx=j*batch_size
            end_inx=(j+1)*batch_size
            C[len(X),start_inx:end_inx] = ssim(torch.tensor(np.ones((224,224,1))), torch.tensor(Y[start_inx:end_inx]))
        if len(gallery)%batch_size!=0:
            start_inx=batch_gallery*batch_size
            C[i,start_inx:-1]=ssim(torch.tensor(np.ones((224,224,1))), torch.tensor(Y[start_inx:]))
        for j in range(len(Y)):
            threshold_Y = C[len(X), j]
            if torch.min(C[:-1,j])>threshold_Y:
                C[:-1,j] = 0
        C[len(X), len(Y)] = ssim(torch.tensor(np.ones((224,224,1))),torch.tensor(np.zeros((224,224,1))),batch=False)
    dirs = os.path.dirname(index)
    if not os.path.exists(dirs): 
        os.makedirs(dirs)
    h = h5py.File(index,'a')
    h.create_dataset('index',data=C)
    h.close()

    # end = time.time()
    # print('------end_time-------',end)
    print('------------------------------------------\n'
          '       exacute_optimal_transport          \n'
          '------------------------------------------')
    gamma = float(0.1)  # 正则化参数
    epsilon = float(1e-2)  # 收敛精度
    reg = float(1e-3)
    a = np.ones((len(X)+1,)) / len(X)+1  # 源域的块概率分布
    b = np.ones((len(Y)+1,)) / len(Y)+1  # 目标域的块概率分布
    alpha= ot.bregman.sinkhorn(a, b, C,reg)  # 这里使用了ot.sinkhorn()函数
    if np.max(alpha[:-1,:-1])>0:
        mapping = np.argmax(alpha[:-1,:-1], axis=1)
    else:
        mapping = np.argmax([],dtype=int)   
    return mapping

if __name__ == '__main__':
    args = parse_args()
    # index_gallery = '/media/whut_zhu/LENOVO_USB_HDD/code/slide_data/A/train_data/Feature256/AtrainMA.h5'
    # index_query = '/media/whut_zhu/LENOVO_USB_HDD/code/slide_data/B/train_data/Feature256/BtrainMA.h5'
    index_gallery = args.gallery
    index_query = args.query
    h5f = h5py.File(index_gallery,'r') # open this file.h5

    f = h5py.File(index_query,'r')
    gallery = h5f['features'] # get features
    query = f['features']
    gallery = np.array(gallery)
    query = np.array(query)
    print('gallery_shape:',gallery.shape)
    Y = np.reshape(gallery,(len(gallery),224,224,1)) #源域
    X = np.reshape(query,(len(query),224,224,1))   #目标域
    print('X_shape:',X.shape)
    print('Y_shape:',Y.shape)
    # mapping=optimal_transport(query,gallery)
# 计算源域和目标域之间的距离矩阵
# C = ot.dist(X, Y)
# 计算源域和目标域之间的距离矩阵，采用SSIM作为距离度量
    # C = np.zeros((len(X)+1, len(Y)+1))


    C = np.zeros((len(X), len(Y)))
    start = time.time()
    print('------start_time-------',start,flush=True)
    batch_size = 6000  # 受限于内存
    if (len(Y) < batch_size):  # 源域的patches数量较小，不会发生溢出
            for i in range(len(X)):
                C[i, 0:len(Y)] = ssim(torch.tensor(X[i]), torch.tensor(Y)).cpu().numpy()
            # end = time.time()
            # print('------end_time-------',end,flush=True)
    else:  # 源域的patches数量较大，会发生溢出需批量处理
            batch_gallery = len(Y) // batch_size
            for i in range(len(X)):
                # C[i, len(Y)] = ssim(torch.tensor(X[i]), torch.tensor(np.ones((224, 224, 1))), batch=False)
                # threshold_X = C[i, len(Y)]
                for j in range(batch_gallery):
                    start_inx = j * batch_size
                    end_inx = (j + 1) * batch_size
                    C[i, start_inx:end_inx] = ssim(torch.tensor(X[i]), torch.tensor(Y[start_inx:end_inx])).cpu().numpy()
                if len(Y) % batch_size != 0:
                    start_inx = batch_gallery * batch_size
                    C[i, start_inx:] = ssim(torch.tensor(X[i]), torch.tensor(Y[start_inx:])).cpu().numpy()
    # for i in range(len(X)):
    #     C[i, len(Y)]=ssim(torch.tensor(X[i]), torch.tensor(np.ones((224, 224, 1))),batch=False)
    #     threshold_X = C[i, len(Y)]
    #     # print('speciali:',threshold_X,flush=True)
    #     C[i,0:len(Y)] = ssim(torch.tensor(X[i]), torch.tensor(Y)).cpu().numpy()
    #     # print('C[i,0]:',C[i,0])
    #     if np.min(C[i,:-1])>threshold_X:
    #         C[i,:-1] = 0
    # C[len(X), 0:len(Y)]=ssim(torch.tensor(np.ones((224,224,1))), torch.tensor(Y)).cpu().numpy()
    # middle = time.time()
    # print('------middle_time-------',middle,flush=True)
    # for j in range(len(Y)):
    #     threshold_Y = C[len(X), j]
    #     # print("specialj:", threshold_Y,flush=True)
    #     if np.min(C[:-1,j])>threshold_Y:
    #         C[:-1,j] = 0
    # C[len(X), len(Y)] = ssim(torch.tensor(np.ones((224,224,1))),torch.tensor(np.zeros((224,224,1))),batch=False)
    # threshold = C[len(X), len(Y)]
    # index = args.cindex
    # end = time.time()
    # print('------end_time-------',end,flush=True)
    # # index = '/home/whut_zhu/save_file/outC/A2B.h5'
    # dirs = os.path.dirname(index)
    # if not os.path.exists(dirs): 
    #     os.makedirs(dirs)
    # h = h5py.File(index,'a')
    # h.create_dataset('index',data=C)
    # h.close()
  
    Y = np.array(Y)
# C = ssim_distance(X_r,Y_r)
# 使用Sinkhorn算法求解最优传输问题
    gamma = float(0.1)  # 正则化参数
    epsilon = float(1e-2)  # 收敛精度
    reg = float(1e-3)
    a = np.ones((len(X),)) / len(X)  # 源域的块概率分布
    b = np.ones((len(Y),)) / len(Y)  # 目标域的块概率分布
# a = np.ones((len(X),))  # 源域的块概率分布
# b = np.ones((len(Y),))  # 目标域的块概率分布
    alpha= ot.bregman.sinkhorn(a, b, C,reg)  # 这里使用了ot.sinkhorn()函数
# alpha = ot.sinkhorn2(a,b,C,gamma,reg)
# 获取从X到Y的最佳映射
    if np.max(alpha[:-1,:-1])>0:
        mapping = np.argmax(alpha[:-1,:-1], axis=1)
    else:
        mapping = np.argmax([],dtype=int)

# 输出结果
    print("Mapping from source domain to target domain:")
    index = args.index
    dirs = os.path.dirname(index)
    if not os.path.exists(dirs): 
        os.makedirs(dirs)
    h = h5py.File(index,'a')
    h.create_dataset('index',data=mapping)
    h.close()
    # print(X)
    # print(Y)
    # print(X[0])
    # print(Y[mapping[0]])
    # print(X[1])
    # print(Y[mapping[1]])
    # print(mapping)
    # index = '/home/whut_zhu/save_file/out/A2B.h5'
    # index = args.index
    # dirs = os.path.dirname(index)
    # if not os.path.exists(dirs): 
    #     os.makedirs(dirs)
    # h = h5py.File(index,'a')
    # h.create_dataset('index',data=mapping)
    # h.close()

    h5f = h5py.File(args.gallery,'r')
    gallery_path = h5f['paths'][:] 
    train_list = []
    for i in range(len(mapping)):
        images_path = str(gallery_path[mapping[i]])[2:-1]
        # print('image_path:',images_path)
        if(images_path not in train_list):# 这行代码可以实验一下
            train_list.append(images_path)
            # weight_list.append(weight[i])
            print('image_path:',images_path)
            # print('weight:', weight[i])
    # train_list.sort()
    print('train_list_len:',len(train_list))
    np.save(args.selected_result, train_list)

    # for i in range(len(X)):
    #     for j in range (len(Y)):
    #         print(C[i,j])