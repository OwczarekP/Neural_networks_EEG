# Neural network in EEG Classification

### **ABSTRACT**

EEG is a one of the major tools used in detection of pathology – for example it can be used to detect the seizure and epilepsy diagnose. Unfortunately, while EEG is a common practice in diagnostic it still require a trained professional to analyze the result and produce interpretation of the signal. The solution for this problem can be use of Machine Learning approach. Multiple research focused on use of ML for this problem – from most simple like regression to more complicated methods like neural networks. However, most of this research focused on one application of EEG in diagnose, like most popular – pre-ictal classification in seizure detection. This project is focused on four different neural networks which provided good accuracy in seizure classification to check if exist networks that can provide good prediction for another application of EEG classification.

### REQUIREMENTS

numpy==1.22.3
pandas==1.4.1
tensorflow==2.8.1
python-dateutil==2.8.2
pytz==2021.3
six==1.16.0

### ARCHITECTURE

**Multi Layer Perceptron**

Multi-Layer Perceptron (MLP) is the simplest of presented networks. MLP is characterized by at least three layers: input, hidden, and output with a nonlinear activation function. The number of hidden layers can change, the same as the number of neurons (units) in each layer.

This network is based on 4 hidden layers with decreasing number of units. The first one has 300 units, and the next ones: are 100, 50, and 20. All of the hidden layers have ReLU activation. Depending on classification different output layer is used. For multi-classification problems, the output layers have 6 units with SoftMax activation. For binary classification, the output is 2 units with sigmoid activation.

**Convolution Neural Network**

Another architecture is based on Convolutional Neural Network (CNN). While it is usually used in image analysis the CNN has good accuracy in a lot of classification problems, including the EEG data [5].

This architecture is based on the 4 Convolutional 1D layers with Max Pooling between. The kernel size is 3 for Conv1D and pool size of 2 in each Pooling layer. The input layer has 32 filters, the next 16, the next 32 with the last Conv1D layer with 64 filters.

**Bi-directional LSTM Network**

Bi-directional Long-Short Term Memory (Bi-LSTM) is a recurrent neural network (RNN), which, compared to LSTM is characterized by having two independent RNNs working together, with allowance to putting information both backward and forward. Thanks to that, Bi-LSMT can preserve information from both the past and the future. Bi-LSTM is used commonly in Natural Language Processing.

The Bi-LSTM network needed additional regularization layers – which were provided in the article [4]. The first Dropout layer with 10% factor was applied before the Bi-LSTM layer and the Dropout layer with a factor of 50% was applied after Bi-LSTM. A bidirectional layer was chosen with the 20 units.

**Convolution Neural Network + Bi-directional LSTM Network**

CNN + Bi-LSTM network was based on the previous CNN and Bi-LSTM architecture. First, the data go through CNN layers, with the same number of filters and pool size as mentioned previously, then through Bi-LSTM layers with the intact Dropout layer and number of units.

All changes in networks were based on the classification of seizure and non-seizure EEG. For other applications even if the network was interpreted as overfitting there were no changes in architecture – the purpose of this project is to find the best architecture for all of the mentioned applications, so the changes in only one application would eliminate the purpose of this project.

RESULTS

***Table 1.** The training and validation accuracy of networks for different type of application*

| Type of classification | MLP | CNN | Bi-LSTM | CNN + Bi-LSTM |
| --- | --- | --- | --- | --- |
| Train | Vali | Train | Val | Train |
| Seizure | 0.9760 | 0.9594 | 0.9711 | 0.9572 |
| Eyes | 0.8331 | 0.7663 | 0.8072 | 0.8243 |
| Tumor | 0.5802 | 0.5851 | 0.5473 | 0.5688 |
| Multi | 0.4754 | 0.4645 | 0.6830 | 0.6500 |

***Table 2.** Test accuracy of networks for different type of application*

| Type of classification | MLP | CNN | Bi-LSTM | CNN + Bi-LSTM |
| --- | --- | --- | --- | --- |
| Seizure | 0.954 | 0.966 | 0.41 | 0.548 |
| Eyes | 0.769 | 0.793 | 0.535 | 0.485 |
| Tumor | 0.519 | 0.501 | 0.502 | 0.49 |
| Multi | 0.467 | 0.662 | 0.178 | 0.193 |

### REFERENCES

1. Hosseini, M., Hosseini, A. and Ahi, K., 2021. A Review on Machine Learning for EEG Signal Processing in Bioengineering. *IEEE Reviews in Biomedical Engineering*, 14, pp.204-218.
2. Kaggle.com. 2022. *Epileptic Seizure Recognition*. [online] Available at: <https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition> [Accessed 26 June 2022].
3. Andrzejak, R., Lehnertz, K., Mormann, F., Rieke, C., David, P. and Elger, C., 2001. Indications of nonlinear deterministic and finite-dimensional structures in time series of brain electrical activity: Dependence on recording region and brain state. *Physical Review E*, 64(6).
4. Daoud, H. and Bayoumi, M., 2019. Efficient Epileptic Seizure Prediction Based on Deep Learning. *IEEE Transactions on Biomedical Circuits and Systems*, 13(5), pp.804-813.
5. Craik, A., He, Y. and Contreras-Vidal, J., 2019. Deep learning for electroencephalogram (EEG) classification tasks: a review. *Journal of Neural Engineering*, 16(3), p.031001.
6. Gemein, L., Schirrmeister, R., Chrabąszcz, P., Wilson, D., Boedecker, J., Schulze-Bonhage, A., Hutter, F. and Ball, T., 2020. Machine-learning-based diagnostics of EEG pathology. *NeuroImage*, 220, p.117021.