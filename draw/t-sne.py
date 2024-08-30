import numpy as np
import pickle
import numpy

# loader1=np.load('/home/zxl/下载/ntu60/xsub10/train_position.npy')
# loader2=np.load('/home/zxl/下载/ntu60/xsub10/train_label.npy')
# label1=open('/home/zxl/下载/ntu60/xsub/train_label.pkl','rb')
# label2=open('/home/zxl/下载/ntu60/xsub10/train_label.pkl','rb')
# s1=pickle.load(label1)
# s2=pickle.load(label2)
# s3=tuple(s2)
# with open('/home/zxl/下载/ntu60/xsub10/train_label.pkl','wb') as f:
#     pickle.dump(s3,f)
# label = np.array(pickle.load(label1))
# # label3 = np.array(pickle.load(label2))
# data1=[]
# data12=[]
# data2=[]
# for i in range(len(label[0])):
#     ccc, l = label[:, i]
#     cc=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#     if l in cc:
#         data1.append(l)
#         data12.append(ccc)
#         data2.append(loader1[i])
#     else:
#         continue
#
# c1=np.array(data1)
# c2=np.array(data2)
# c3=np.array(data12)
# c4=(data12,data1)
# np.save('/home/zxl/下载/ntu60/xsub10/train_position.npy',c2)
# with open('/home/zxl/下载/ntu60/xsub10/train_label.pkl','wb') as f:
#     pickle.dump(c4,f)
#     # pickle.dump(c1.f)
# print()

# loader1=np.load('/home/zxl/下载/ntu60/xsub10-best/val_position.npy')
# loader2=np.load('/home/zxl/下载/ntu60/xsub10-best/val_label.npy')
# data1=[]
# data2=[]
# for i in range(len(loader2)):
#     j=loader2[i]
#     if 0<=j<10:
#         data1.append(loader2[i])
#         data2.append(loader1[i])
#     else:
#         continue
#
# c1=np.array(data1)
# c2=np.array(data2)
#
# np.save('/home/zxl/下载/ntu60/xsub10-best/val_position10.npy',c2)
# np.save('/home/zxl/下载/ntu60/xsub10-best/val_label10.npy',c1)

loader1=np.load('/home/zxl/下载/ntu60/xsub10-best/val_position60-scale.npy')  
loader2=np.load('/home/zxl/下载/ntu60/xsub10-best/val_label60-scale.npy')
data0=[]
data1=[]
data2=[]
data3=[]
data4=[]
data5=[]
data6=[]
data7=[]
data8=[]
data9=[]
for i in range(len(loader2)):
    j=loader2[i]
    if j==8:
        data0.append(loader1[i])
    if j==13:
        data1.append(loader1[i])
    if j==14:
        data2.append(loader1[i])
    if j==17:
        data3.append(loader1[i])
    if j==25:
        data4.append(loader1[i])
    if j==35:
        data5.append(loader1[i])
    if j==48:
        data6.append(loader1[i])
    if j==49:
        data7.append(loader1[i])
    if j==41:
        data8.append(loader1[i])
    if j == 58:
        data9.append(loader1[i])
    else:
        continue
c0=np.array(data0)[0:200]
c1=np.array(data1)[0:200]
c2=np.array(data2)[0:200]
c3=np.array(data3)[0:200]
c4=np.array(data4)[0:200]
c5=np.array(data5)[0:200]
c6=np.array(data6)[0:200]
c7=np.array(data7)[0:200]
c8=np.array(data8)[0:200]
c9=np.array(data9)[0:200]

im_k_motion = np.zeros((2000,60))

for i in range(200):
    c00=c0[i]
    c11=c1[i]
    c22 = c2[i]
    c33 = c3[i]
    c44 = c4[i]
    c55 = c5[i]
    c66 = c6[i]
    c77 = c7[i]
    c88 = c8[i]
    c99 = c9[i]
    c10=np.row_stack((c00,c11,c22,c33,c44,c55,c66,c77,c88,c99))
    j=(i+1)*10
    im_k_motion[i*10:j,:]=c10
# s1=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,
#              30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59])
# ss=np.expand_dims(s1,axis=1)
# im_q_motion = np.ones((1,2000))
# s2=numpy.dot(ss,im_q_motion).T
# s2=s2.reshape(-1)
np.save('/home/zxl/下载/ntu60/xsub10-best/val_position60-scale1.npy',im_k_motion)
# np.save('/home/zxl/下载/ntu60/xsub10-best/val_label60.npy',s2)

# loader1=np.load('/home/zxl/下载/ntu60/xsub10-best/val_position0-9.npy')
# loader2=np.load('/home/zxl/下载/ntu60/xsub10-best/val_position10-19.npy')
# loader3=np.load('/home/zxl/下载/ntu60/xsub10-best/val_position20-29.npy')
# loader4=np.load('/home/zxl/下载/ntu60/xsub10-best/val_position30-39.npy')
# loader5=np.load('/home/zxl/下载/ntu60/xsub10-best/val_position40-49.npy')
# loader6=np.load('/home/zxl/下载/ntu60/xsub10-best/val_position50-59.npy')
#
# im_k_motion = np.zeros((12000,60))
# #
# for i in range(200):
#     c00=loader1[10*i:(i+1)*10]
#     c11=loader2[10*i:(i+1)*10]
#     c22 = loader3[10*i:(i+1)*10]
#     c33 = loader4[10*i:(i+1)*10]
#     c44 = loader5[10*i:(i+1)*10]
#     c55 = loader6[10*i:(i+1)*10]
#     c10=np.row_stack((c00,c11,c22,c33,c44,c55))
#     im_k_motion[60*i:(i+1)*60,:]=c10
# np.save('/home/zxl/下载/ntu60/xsub10-best/val_position60.npy',im_k_motion)
from sklearn import manifold,datasets
import time
import numpy as np
import matplotlib.pyplot as plt

loader1=np.load('/home/zxl/下载/ntu60/xsub11-best/val_position-1.npy')
loader2=np.load('/home/zxl/下载/ntu60/xsub10-best/val_label10.npy')
loader2=np.append(loader2,loader2)
# %%
n_components = 2


# %%
digits = datasets.load_digits(n_class=10)
data = loader1 
label = loader2 
n_samples, n_features = data.shape  

# %%
tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0, perplexity=30)
start = time.time()
result = tsne.fit_transform(data)
end = time.time()
print('t-SNE time: {}'.format(end-start))

# %%
# result
# cora_label_to_color_map = {0: "red", 1: "blue", 2: "green", 3: "orange", 4: "yellow", 5: "pink", 6: "gray",7: "red", 8: "blue", 9: "green"}
cora_label_to_color_map = {0: "red", 1: "blue", 2: "green", 3: "orange", 4: "yellow", 5: "pink", 6: "gray",7: "red", 8: "blue", 9: "green"}

# %%
x_min, x_max = np.min(result, 0), np.max(result, 0)  
result = (result-x_min)/(x_max-x_min) 
ax = plt.subplot(111)  
for i in range(n_samples):
    if label[i]==0:
        colorplt="lightcoral"
        shape='o'
    if label[i]==1:
        colorplt="coral"
        shape ='s'
    if label[i]==2:
        colorplt="darkorange"
        shape ='p'
    if label[i]==3:
        colorplt="gold"
        shape ='+'
    if label[i]==4:
        colorplt="palegreen"
        shape ='h'
    if label[i]==5:
        colorplt="pink"
        shape ='v'
    if label[i]==6:
        colorplt="skyblue"
        shape ='*'
    if label[i]==7:
        colorplt="paleturquoise"
        shape ='x'
    if label[i]==8:
        colorplt="hotpink"
        shape ='d'
    elif label[i]==9:
        colorplt="plum"
        shape ='^'
    # plt.text(result[i, 0], result[i, 1], str(label[i]), color=plt.cm.Set1(label[i] / 10.), fontdict={'weight': 'bold', 'size': 9})
    plt.scatter(result[i, 0], result[i, 1], color=colorplt, marker='o',s=15)
plt.xticks([]) 
plt.yticks([])
plt.title('t-SNE(SkeletonCLR)',fontsize=15)
plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# import json
# from sklearn import manifold
#
#
# cora_label_to_color_map = {0: "red", 1: "blue", 2: "green", 3: "orange", 4: "yellow", 5: "pink", 6: "gray"}
#
#
# def draw_tsne(loade1,loade2):
#     X = loade1
#     y = loade2
#     '''t-SNE'''
#     tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
#     X_tsne = tsne.fit_transform(X)
#     print(X.shape)
#     print(X_tsne.shape)
#     print(y.shape)
#     print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
#
#     '''嵌入空间可视化'''
#     x_min, x_max = X_tsne.min(0), X_tsne.max(0)
#     X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
#     plt.figure(figsize=(8, 8))
#     for i in range(X_norm.shape[0]):
#         plt.text(X_norm[i, 0], X_norm[i, 1], '*', color='type',
#                  fontdict={'weight': 'bold', 'size': 18})
#     plt.xticks([])
#     plt.yticks([])
#     plt.show()
#
#
# draw_tsne(loader1,loader2)

