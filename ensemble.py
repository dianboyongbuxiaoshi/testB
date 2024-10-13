import argparse
import pickle

import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', default='ntu/xview', choices={'kinetics', 'ntu/xsub', 'ntu/xview'},
                    help='the work folder for storing results')
parser.add_argument('--alpha', default=1.05, help='weighted summation')
arg = parser.parse_args()

dataset = arg.datasets
#label = open('./data/' + dataset + '/val_label.pkl', 'rb')
#label = np.array(pickle.load(label))
label=np.load('E:/骨骼数据/2s-AGCN-master/data/ntu/xview/test_label_A.npy')
r1 = open('./work_dir/' + dataset + '/agcn_test_joint929/epoch1_test_score.pkl', 'rb')
#r1 = list(pickle.load(r1).items())
r1=pickle.load(r1)
#r2 = open('./work_dir/' + dataset + '/agcn_test_bone917/epoch1_test_score.pkl', 'rb')
r2 = open(r'E:/骨骼数据/TE-GCN-main/work_dir/ntu/xview/test_joint919/epoch1_test_score.pkl', 'rb')
#r2 = list(pickle.load(r2).items())
r2=pickle.load(r2)
right_num = total_num = right_num_5 = 0
for i in tqdm(range(label.shape[0])):
    l = label[i]
    r11 = r1[i]
    r22 = r2[i]
   #_, l = label[:, i]
   # _, r11 = r1[i]
   # _, r22 = r2[i]
    r = r11 + r22 * arg.alpha  #先在这里进行最终精确度计算，如何精确度有所提高，就按照此得分融合方式进行pkl文件的相加
    print(r.shape)
    rank_5 = r.argsort()[-5:]
    right_num_5 += int(int(l) in rank_5)
    r = np.argmax(r)
    right_num += int(r == int(l))
    total_num += 1
acc = right_num / total_num
acc5 = right_num_5 / total_num
print(acc, acc5)
