# Multiple Article Classification NLP

## NLP (Natural Language Processing) - Machine Learning: 

This project, NLP were used on multiple article classification. Analysing data from  unseen articles in text format, and categorize into 5 categories namely Sport, Tech, Business, Entertainment and Politics using machine learning approach.

## LSTM

3 layers of LSTM (Long short term memory) model were chosen from RNN (Recurrent Neural Networks) since it capable of learning long-term dependencies, especially in sequence prediction problems

![model](https://user-images.githubusercontent.com/106498393/180982290-4bfc890d-199d-4a16-8071-f8c17de976f8.png)

## Accuracy >90%

I've tried different random_state error and epoch values to get the best accuracy with good model (without overfitting nor underfitting). The best model by far is when using random_state = 1, epoch <7 or applying Earlycallback functions.

Presented here is f1 accuracy score at 0.96 = 96%.
Noted that accuracy score on predict model also got 0.96%

<img width="510" alt="mac_accuracy" src="https://user-images.githubusercontent.com/106498393/180982991-9b40a15c-fee9-43b0-8c88-001ace6b4800.png">

## Tensorboard
Snippet from tensorboard.

<img width="1094" alt="mac_tensorboard" src="https://user-images.githubusercontent.com/106498393/180983546-33de7a08-90c6-457c-b48d-863fd373d9c2.png">

## Data Source
Thanks to this awesome [data](https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv) 

## Execution
There are 3 .py file included in this repo:
* mac_train
* mac_module
* mac_deploy

To test the model, Google Colab link has been added in file : 
      
      `mac_train

you can just execute using Colab after include mac_module in the Colab file.

To test out, try the mac_deploy file as well. Have fun!


## Contributing

:heart: Firstly, thank you for taking the time to contribute to this project! :heart:

Steps to contribute:
* Make your awesome changes
* Submit pull request; if you add a new entry, please give a very brief explanation why you think it should be added.

![Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

