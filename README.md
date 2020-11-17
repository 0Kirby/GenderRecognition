# GenderRecognition
江南大学《机器学习》大作业-人脸图像性别分类
# 简介
[Kaggle比赛地址](https://www.kaggle.com/c/jiangnan2020)  
[大报告](https://blog.csdn.net/jty123456/article/details/109733826)  
[Jupyter Notebook实现](https://blog.csdn.net/jty123456/article/details/109666045)  
**《机器学习》** 是计算机类硕士生阶段的重要应用基础课程，旨在使学生了解机器学习的研究对象、研究内容、研究方法（包括理论基础和实验方法），和近年来的主要发展技术路线。本课程非学位课，为了培养学生以机器学习理论为基础从事相关应用研究的能力，学生的考核以大作业的形式进行。
通过看——查阅资料、做——复现已有资料的实验或做一个机器学习应用课题、写——将自己的研究工作写成技术报告，完成整个大作业。 
# 库
|  名称  |   版本   |  用途  |
|  :----:  |  :----:  |  :----:  |
|  Tensorflow |  2.3.1  |  深度学习框架  |
|  Keras |  2.4.3  |  基于Tensorflow的实现  |
|  scikit-learn  |  0.32.2  |  机器学习库  |
|  matplotlib |  3.3.2  |  绘图库  |
|  pandas  |  1.1.3  |  数据处理库  |
|  numpy  |  1.19.2  |  矩阵库  |
|  opencv-Python  |  4.4.0.44  |  读取图片  |   
# 数据集
[从Kaggle下载](https://www.kaggle.com/c/jiangnan2020/data)  
然后把下载到的所有文件放在**根目录**
# 文件说明
GenderRecognition.ipynb - 包含运行结果的交互式Jupyter Notebook  
run.py - 纯Python代码  
save_weights.h5 - 训练2000轮后的权重，可复现最佳预测结果  
train.csv - 已对其中错误的标签进行修改  
submission.csv - 输出的预测结果  
trial.txt - 使用最佳模型预测训练集，对训练集所做的修改  
Jupyter Notebook Preview.html/Jupyter Notebook Preview.pdf - 运行结果，内容同GenderRecognition.ipynb

