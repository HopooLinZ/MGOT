import argparse

parser = argparse.ArgumentParser(description='FIDTM')

parser.add_argument('--dataset', type=str, default='zhl',
                    help='choice dataset')
parser.add_argument('--save_path', type=str, default='save_file/zhl',
                    help='save checkpoint directory')
parser.add_argument('--path_save', type=str, default='/public/home/qiuyl/CCluster/save/B2B_feature_vis',
                    help='save generated pseudo dataset directory')
parser.add_argument('--workers', type=int, default=16,
                    help='load data workers')
parser.add_argument('--print_freq', type=int, default=25,
                    help='print frequency')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='start epoch for training')
parser.add_argument('--epochs', type=int, default=2000,
                    help='end epoch for training')
parser.add_argument('--pre', type=str, default='/HOME/scw6cs4/run/FIDT_main/checkpoint/model_best_57.pth',
                    help='pre-trained model directory')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--crop_size', type=int, default=256,
                    help='crop size for training')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--best_pred', type=int, default=1e7,
                    help='best pred')
parser.add_argument('--gpu_id', type=str, default='0',
                    help='gpu id')
parser.add_argument('--lr', type=float, default= 1e-5,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5 * 1e-4,
                    help='weight decay')
parser.add_argument('--preload_data', type=bool, default=True,
                    help='preload data. ')
parser.add_argument('--visual', type=bool, default=False,
                    help='visual for bounding box. ')

'''video demo'''
parser.add_argument('--video_path', type=str, default=None,
                    help='input video path ')

args = parser.parse_args()
return_args = parser.parse_args()
