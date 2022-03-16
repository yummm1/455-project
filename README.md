# 455-project
## Summary
Used resnet to do blood cell image classification. Applied transfer learning and draw loss decreasing graph and accuracy increasing graph. The best accuracy is 0.714, which is an acceptable result. 

## Problem Setup
Blood smear is one of the most common test that doctors perform, and it is very useful in detecting disease and distinguish diseases. The computer-run clarification would greatly save the cost and is more time-efficient. Machine learning and computer vision can be applied in blood cell clarification. 
The goal of this project is to clarify 4 kinds of white blood cells that could indicate different health conditions. 

## Dataset: what is the data that you used, where did you get it, why did you choose it, did you have to do anything to wrangle it?
The data used is 336 blood cells images that include neutrophil, lymphocyte, eosinophil, monocyte and basophil. The data is already cleaned and output into csv file as their path to image and their labels. The original images can be found on Kaggle, and the cleaned version is online. I chose it because it the properly labeled and the dataset is a proper number. Even though the number of images in each category is unbalanced, the dataset is rebuilt by using images from the same category for multiple times. 

## Techniques: 
Model choosing: Resnet34<br>
1. Resnet34 solves degradation problem, which gives best result on the classification task. It is pretrained model with high precision. It is also a relatively new model but has good community environment
2. Residual learning framework helps with the training of deep neural network by solves degradation problem. It uses shortcut to realize the identity mapping
<br>
Training parameters:<br>
1. Optimizer: Adam <br>
2. Loss function: Cross entropy loss <br>
3. Number of training epochs: 15 <br>
4. Learning rate: 0.01 (by default) <br>
5. Momentum: 0.1(by default)
<br>
<br>
Evaluation metrics:<br>
1.	Loss decreasing graph (Both training set and validation set)<br>
2.	Accuracy increasing graph (Both training set and validation set)<br>
3.	Accuracy for holdout set

## Additional Info: Anything else needed to fully describe your work
Due to the time limited, we don’t find the best hyper-parameters combination. If we got time, we could try another loss functions like the negative log likelihood loss and traditional loss like the mean absolute error loss or the mean squared logarithmic error loss. We also can train more epochs and test for more model like efficient net and AlexNet.

## References: 
He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). https://doi.org/10.1109/cvpr.2016.90 <br>

Home Page. RESNET. (2021, July 21). Retrieved March 15, 2022, from https://www.resnet.us/ <br>

【图像分类】实战——使用ResNet实现猫狗分类（pytorch）_AI小浩的技术博客_51CTO博客. (n.d.). Blog.51cto.com. Retrieved March 16, 2022, from https://blog.51cto.com/AIXhao/2996855
