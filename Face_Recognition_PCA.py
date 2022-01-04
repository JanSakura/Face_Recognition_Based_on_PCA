#导入包，numpy高性能矩阵计算
import numpy as np
from PIL import Image
import os     #读取文件夹的
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt     #pycharm 社区版的问题，不能显示plt.show()的图像,必须要专业版
import cv2
#提取图片数据
#plt_show显示灰度图片，灰度图处理速度会快一点
def plt_show(img):
    plt.imshow(img,cmap='gray')
    plt.show()

#读取一个文件夹下的所有图片，输入参数是文件名，返回文件地址列表
def read_directory(directory_name):
    faces_addr = []
    for filename in os.listdir(directory_name):     #读取文件夹下面的文件夹名
        faces_addr.append(directory_name + "/" + filename)
    return faces_addr

#读取所有人脸文件夹,保存图像地址在列表中
faces = []
for i in range(1,42):   #从s1到s41，s41的人脸来自自己
    faces_addr = read_directory('C://Picture/orl_faces/s'+str(i))
    for addr in faces_addr:
        faces.append(addr)

#读取图片数据,生成列表标签
images = []                #图片列表
labels = []                #做一个标签列表
for index,face in enumerate(faces):
    image = cv2.imread(face,0)
    images.append(image)
    labels.append(int(index/10+1))
print("输出图片数量和带标签的图片数量:")
print(len(labels),len(images))
print("输出类型和图片标签的具体情况:")
print(type(images[0]))
print(labels)

#为了省时间,只画出最后两组人脸图像
#创建画布和子图对象
fig, axes = plt.subplots(2,10
                       ,figsize=(15,4)
                       ,subplot_kw = {"xticks":[],"yticks":[]} #不要显示坐标轴
                       )
#填充图像
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i+390],cmap="gray") #选择色彩的模式,下面的都是灰色,节省运行时间

#PCA降维
#图像数据转换特征矩阵
#图片都是当作向量来处理的，而每个图片的向量维度大小都是92*112，是非常大的维度，所以要降维
image_data = []       #每一个图片都是高112，宽92的数组，我们把92*112个数据放到一个列表里，10304个像素点
for image in images:
    data = image.flatten()
    image_data.append(data)
#print("输出图片维度数:")
#print(image_data[0].shape)

#转换为numpy数组，就是个（410，10304）的数组，即410个图片，每个图10304维
X = np.array(image_data)
y = np.array(labels)
print("输出数据类型,和图片个数与维度")
print(type(X))
print(X.shape)

# 导入sklearn的pca模块
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# 画出特征矩阵，panel data（面板数据）和 Python data analysis（Python 数据分析）
import pandas as pd
data = pd.DataFrame(X)
data.head()

# 划分数据集，分别为训练数据集，训练测试集，训练标签，测试标签。测试比例0.2，328个训练集，82个测试集
x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.2)

# 训练PCA模型，降到100维度
pca=PCA(n_components=100)
pca.fit(x_train)

# 返回测试集和训练集降维后的数据集
x_train_pca = pca.transform(x_train)  #得到我们降维的数据
x_test_pca = pca.transform(x_test)    #得到测试集降维的数据，也是100维度
print("输出训练集数量和测试集数量,以及降维后的维度:")
print(x_train_pca.shape,x_test_pca.shape)

#画出来，（100，10304）100个特征，10304维度
V = pca.components_
V.shape

# 100个特征脸
#创建画布和子图对象
fig, axes = plt.subplots(10,10
                       ,figsize=(15,15)
                       ,subplot_kw = {"xticks":[],"yticks":[]} #不要显示坐标轴
                       )
#填充图像
for i, ax in enumerate(axes.flat):
    ax.imshow(V[i,:].reshape(112,92),cmap="gray") #选择色彩的模式

# 改选择多少个特征呢？
#属性explained_variance_ratio，查看降维后每个新特征向量所占的信息量占原始数据总信息量的百分比
#又叫做可解释方差贡献率，即100个特征在原图中的贡献率，加起来不到1，因为我们减少了一些维度
pca.explained_variance_ratio_

# 返回特征所携带的数据是原始数据的多少
pca.explained_variance_ratio_.sum()

# 画出特征个数和所携带信息数的曲线图
explained_variance_ratio = []
for i in range(1,151):         #保留1个特征到151个特征的信息量
    pca=PCA(n_components=i).fit(x_train)
    explained_variance_ratio.append(pca.explained_variance_ratio_.sum())
plt.plot(range(1,151),explained_variance_ratio)
plt.show()
print(matplotlib.get_backend())

#使用OpenCV的EigenFace算法进行识别
#原理：将训练集图像和测试集图像都投影到特征向量空间中，再使用聚类方法（最近邻或k近邻等）得到里测试集中的每个图像最近的图像，进行分类即可。
#cv2.face.EigenFaceRecognizer_create()创建人脸识别的模型，通过图像数组和对应标签数组来训练模型
#模型model创建与训练model
model = cv2.face.EigenFaceRecognizer_create()
model.train(x_train,y_train)

#预测,第几个人的置信度,越小越匹配,最好为0
res = model.predict(x_test[0])
print("输出测试集第0个的标签,和其置信度:")
print(res)

y_test[0]        #测试集第0个人

# 测试数据集的准确率
ress = []
true = 0
for i in range(len(y_test)):
    res = model.predict(x_test[i])
#如果识别错误print(res[0]),打印出来
    if y_test[i] == res[0]:
        true = true+1
    else:
        print("识别错误的图片标签:")
        print(i)

print('测试集识别准确率：%.2f'% (true/len(y_test)))

#平均脸,也就是所有图片生成的脸,下面每一个数据都是各降维后的每个像素点累加和
mean = model.getMean()
print(mean)
meanFace = mean.reshape(112,92)
plt_show(meanFace)

#人脸识别测试
#降维
pca=PCA(n_components=100)
pca.fit(X)
X = pca.transform(X)

# 将所有数据都用作训练集
# 模型创建与训练
model = cv2.face.EigenFaceRecognizer_create()
model.train(X,y)

# plt显示彩色图片
def plt_show0(img):
    b,g,r = cv2.split(img)
    img = cv2.merge([r, g, b])
    plt.imshow(img)
    plt.show()

# 输入图片识别
#img = cv2.imread('C://Picrure/orl_faces/pic1.jpg')
# 灰度处理
img = cv2.imread('C://Picture/orl_faces/pic1.jpg',0)

imgs = []
imgs.append(img)

# 特征矩阵
image_data = []
for img in imgs:
    data = img.flatten()
    image_data.append(data)

test = np.array(image_data)

# 用训练好的pca模型给图片降维
test = pca.transform(test)
test[0].shape

res = model.predict(test)
res
print('人脸识别结果：',res[0])

#Opencv中简单的人脸检测
#加载人脸检测模型,这是opencv里有的直接用
face_engine = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
img = cv2.imread('C://Picture/orl_faces/pic2.png')
plt_show0(img)

#复制图像灰度处理
img_ = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#检测人脸获取人脸区域
faces = face_engine.detectMultiScale(gray)

# 将检测出的人脸可视化,w为宽,h为高.默认点是左上角的点
for(x, y, w, h) in faces:
    cv2.rectangle(img_, (x, y), (x + w, y + h), (0, 0, 255), 3)
    plt_show0(img_)
    face = img[y:y + w, x:x + h]
    plt_show0(face)
