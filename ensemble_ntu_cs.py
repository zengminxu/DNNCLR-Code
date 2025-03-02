import pickle
import numpy as np
from tqdm import tqdm

# Linear
print('-' * 20 + 'Linear Eval' + '-' * 20)

joint_path = '/home/zhangxuelian/code_wuyushan/wu_aimclr_2.3.0/work_dir/joint_xsub/202212081233/linear_eval/'
bone_path = '/home/zhangxuelian/code_wuyushan/wu_aimclr_2.3.0/work_dir/bone_xsub/202212081233/linear_eval/'
motion_path = '/home/zhangxuelian/code_wuyushan/wu_aimclr_2.3.0/work_dir/motion_xsub/202212081233/linear_eval/'

# label = open('/home/zhangxuelian/code_wuyushan/wu/data/Base/NTU120_frame50/xsub/val_label.pkl', 'rb')
label = open('/home/zhangxuelian/code_wuyushan/wu/data/Base/NTU-RGB-D/xsub/val_label.pkl', 'rb')
label = np.array(pickle.load(label))
r1 = open(joint_path + 'test_result.pkl', 'rb')
r1 = list(pickle.load(r1).items())
r2 = open(bone_path + 'test_result.pkl', 'rb')
r2 = list(pickle.load(r2).items())
r3 = open(motion_path + 'test_result.pkl', 'rb')
r3 = list(pickle.load(r3).items())

alpha = [0.6, 0.6, 0.4] # joint, bone, motion

right_num = total_num = right_num_5 = 0
for i in tqdm(range(len(label[0]))):
    _, l = label[:, i]
    _, r11 = r1[i]
    _, r22 = r2[i]
    _, r33 = r3[i]
    r = r11 * alpha[0] + r22 * alpha[1] + r33 * alpha[2]
    rank_5 = r.argsort()[-5:]
    right_num_5 += int(int(l) in rank_5)
    r = np.argmax(r)
    right_num += int(r == int(l))
    total_num += 1
acc = right_num / total_num
acc5 = right_num_5 / total_num
print('top1: ', acc)
print('top5: ', acc5)