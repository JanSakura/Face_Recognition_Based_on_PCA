# Face_Recognition_Based_on_PCA
use OpenCV2 and SKlearn

the face images are from orl_faces databases
添加了文件夹s41,里面是自己新增的同一个人的十张人脸照片,也是92*112,其中后来检测的pic1的人脸,也应是和s41的来自同一个人.

关于有些图标没显示,使用Pycharm Professional Edition,并修改.如matplotlib.pyplot.show()无法显示图片
使用的包：sklearn ，matplotlib ，numpy ，pillow ，OpenCV2 

代码大致流程
1.提取图片数据
2.图片PCA降维
3.对测试集人脸识别
4.数据集之外的人脸图片验证
5.非纯人脸照片的人脸检测（额外的，与本数据集的人脸识别无关,可以不用）
