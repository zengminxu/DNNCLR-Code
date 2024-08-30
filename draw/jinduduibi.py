from matplotlib.font_manager import *
import matplotlib.pyplot as plt
from  matplotlib.pyplot import MultipleLocator
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.figure(figsize=(8, 4), dpi=200)
plt.subplot(111)
# x = np.linspace(0.5, 2, 4, endpoint=True)
x=[0.6,0.8,1,1.2]
x2=[0.6]
y=[39.1,48.1,35.6,39.7]
y1= [52.6]
y2= [50.7,76.3,42.7,41.7]
y3= [58.5,64.8,48.6,49.2]
y4= [68.3,76.4,56.8,55.9]
y5= [71.5,76.5,57.6,54.6]
y6= [70.4,77.9,60.1,62.2]
y7= [70.3,75.2,59.0,63.6]
s1=plt.scatter(x,y,s=550,lw=10,color='plum',marker='^')
s2=plt.scatter(x2,y1,s=550,lw=10,color='paleturquoise',marker='s')
s3=plt.scatter(x,y2,s=550,lw=10,color='gold',marker='D')
s4=plt.scatter(x,y3,s=550,lw=10,color='darkorange',marker='p')
s5=plt.scatter(x,y4,s=550,lw=10,color='skyblue',marker='d')
s6=plt.scatter(x,y5,s=550,lw=10,color='lawngreen',marker='X')
s7=plt.scatter(x,y6,s=550,lw=10,color='hotpink',marker='*')
s8=plt.scatter(x,y7,s=550,lw=10,color='r',marker='o')
# ,,,,,,
# grey,silver,lightgrey,lightgray,darkgrey,slategrey,dimgray
# c = [1, 2, 3, 4, 5, 6, 7, 1, 3, 4, 5, 6,7,1, 3, 4, 5, 6,7,1, 3, 4, 5, 6,7]
# plt.xticks([1,10,20,30,40,50,60,70,80,90,100,110],fontsize=12);
# plt.yticks([t for t in range(0,13,2) ],fontsize=12);
font1 = {'family' : 'simsun',
    'weight' : 'normal',
    'size' : 15,
    }
font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 10,
    }
import matplotlib
print(matplotlib.matplotlib_fname())
myfont = FontProperties(fname='/home/zxl/anaconda3/envs/train/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf')
x_major_locator=MultipleLocator(0.2)
plt.xlim(0.55,1.25)
plt.gca().xaxis.set_major_locator(x_major_locator)
plt.gca().set_xticklabels(['','xsub60','xview60','xsub120','xset120'],fontsize=27)
plt.gca().set_yticklabels(['30','40','50', '60','70','80'],fontsize=27)


plt.legend((s1,s2,s3,s4,s5,s6,s7,s8),('LongT GAN','MS2L','P&C','AS-CAL','SkeletonCLR','SGCLR(ours)','CrosView-SGCLR(ours)','CrosScale-SGCLR(ours)'),loc='best',fontsize=26)
# plt.xlabel(r'Dataset',font1)
# plt.ylabel(r"Accuracy/%",font1)

plt.xlabel('四种基准数据集',fontproperties=myfont,fontsize=30)
plt.ylabel("精确度/%",fontproperties=myfont,fontsize=30)

plt.show()
plt.savefig("figure_1.png",dpi=1, bbox_inches='tight',pad_inches=0.02)
