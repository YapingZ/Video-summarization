# real_time_face_recognition
由于不能满足当前的tensorflow版本，以及未能满足设计要求，进行了优化。  
* **采用facenet作为embedding嵌入模型，而非nn4神经网络**
* **使用原facenet代码的compare的思路进行人脸的比较，放弃了knn分类**
* **实现了无需训练分类模型，实时的比较人脸**

## Workflow
1.python3.6  
2.tensorflow=1.3.0(可运行在无gpu版)

## Running
1.从 https://github.com/davidsandberg/facenet 中下载预训练的分类模型，放在model_check_point下  
2.使用pip install requirements.txt安装需要的包。  
3.在目录下新建picture文件，将需要识别的人的图片放入其中，每人放入一张清晰的图片即可 ,当然也可以是多张图片。 
4.执行python real_time_face_recognize.py 

实现功能：
1. 可以实现实时的检测所需要人脸。

2.可以把出现过此人的帧并保存在保存在output文件夹内。

3.可以定位到此人出现的时间与最后出现的时间，便于生成此人的视频摘要。



Runing：
根据项目需要，依据上面Runing 1， 2, 3
执行python main.py

output2 文件夹是要检测的图片。
功能：对图片进行进行检测，出现校长为1,书记为2 同时出现校长书记为12 未出现两个人为0。


不会因为阈值大小出现几个人被识别成一个人在同一帧上。
 

论文 

1. Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks
2. A Unified Embedding for Face Recognition and Clustering





