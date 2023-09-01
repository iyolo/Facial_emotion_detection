# Facial_emotion_detection

We proposed a modified CoAtNet for emotion classification task. The proposed method outperformed the vanilla CoAtNet and other baseline 
models by close to 9% in terms of accuracy. Architecture modification was crucial, demonstrating the neccesity of retaining fine-grain detail 
for such task. Temperature smoothing introduced additional benefits due to its ability to allow the model to quickly adjust the smoothness of 
the final probability distribution.


# Installation and Usage

You need to install Openpose (https://github.com/CMU-Perceptual-Computing-Lab/openpose) and install the python libraries required in the requirements.txt file.

This paper uses RAF-DB dataset(http://www.whdeng.cn/raf/model1.html).In this experiment, the dataset is divided into training, validation and test sets. This division helps to 
evaluate and optimize the performance model and guarantee the generalization ability of the model. Since the number of pictures is different for each category, a stratified 
sampling method is adopted. There are 10,000 pictures in the training set accounting for 65%, 2271 pictures in the verification set accounting for 15%, and 3068 pictures in the test set accounting for 20%. You should use ProcessImages.ipynb to get processed images.

After installing the required libraries and processing the image, you can start running the project with the command bash run.sh.

# Features and Functionality

Combining deep learning and VIT for human emotion detection, and greatly improving the accuracy of human emotion detection.


# Main results

| Model         | ACC   | F1    |
|---------------|-------|-------|
| ResNet        | 63.01 | 64.79 |
| InceptionV3   | 64.75 | 66.39 |
| CoAtNet (vanilla) | 64.31 | 65.82 |
| CoAtNet (ours) | 73.20 | 73.68 |


# Testing Demo

## 1.Directory settings

Set the corresponding folders in config.py, of which processed_data_folder and train_log_folder need to be created by yourself.

```{python}
self.path.json_path = "/home/yolo/Study/network-model/new_network_training/record.json"
self.path.processed_data_folder = "/home/yolo/Study/network-model/organized"
self.path.train_log_folder = "/home/yolo/Study/network-model/train_logs"
```
## 2.Model settings

```{python}
# change different models(CCCC,CCTT,TTTT) and channels(3,4)
self.train.case_name = "coatnet_CCTT_4"
# change train batch size(16,32,128,256)
self.train.train_batch_size = 256
```
## 3.Model selection

Select different models in the models file to be referenced in basic_supervised_trainer.py.

## 4.Training model

![4J_K9KGD }WVW{9YA1ZI{NS](https://github.com/iyolo/Facial_emotion_detection/assets/49433145/563b0f0f-1777-46be-9a92-dda5d8741b92)

## 5.Model on test set

First, select the model in the model folder in test.py, and then set the path to the model folder trained in 4.

```{python}
from model.coatnet import CoAtNet
test_path = '/home/yolo/network-model/train_logs/2023_Jun_21_PM_08_36_49_coatnet_TTTT_4_keypoints'
```
## 6.Get result

```{r}
json_data <- '{
    "epoch": 45,
    "Accuracy": 0.6000299453735352,
    "F1": 0.5992236137390137
}'
```

# Issues and Feedback
If you encounter any issues or have suggestions, please submit them in GitHub Issues.

# Acknowledgements and Thanks


Dr. Xinfeng Ye’s dedication to my academic and personal growth has been truly remarkable. His extensive knowledge, expertise, and experience in the 
field have provided me with invaluable insights and perspectives. I would also like to thank my supervisor, Mano Manoharan, for his great support
during my study. I also want to express my gratitude to my colleagues for their contribution in helping me understanding the problem and business process.
In conclusion, I want to express my deepest gratitude to my family, especially my parents, for their unconditional love, unwavering support, and belief in me. 
