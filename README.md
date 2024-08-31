# Deep Learning for Glitch Classification in Gravitational Waves: a Comparison of CNN-BiLSTM and CNN-SLP Models
In many fields of physics the gravitational waves are studied to deepen the understanding of the fundamentals of physics and of many astronomic processes. The detection of gravitational waves can be achieved by laser-interferometric detectors, which are sensitive to changes in distance under the scale of atomic nuclei. Although the instruments are isolated from non-astrophysical noise, the detectors are still susceptible to instrumental and environmental noise, which can cause gravitational waves to be affected by phenomena called "glitches" that hinder research.

Building on the Gravity Spy project, which aims to classify glitches affecting gravitational wave detectors into morphological families, this paper proposes a deep learning technique to make a contribution to the accurate classification of these glitch phenomena.
In this work, the Gravity Spy dataset is presented and used for the classification task. The dataset consists of time-frequency images of glitches, organized in 22 classes based on glitch morphology. In particular, for each gravitational wave, the dataset includes a set of four images captured with four different time windows.

In this work we compare two deep learning network designs for glitch classification.
Both approaches consists in an initial set of three CNNs, each processing images associated to a specific time window to extract feature vectors. On the second part, which performs the classification task, the first approach uses a Single-layer Perceptron (SLP) network, while the second approach uses a Bidirectional LSTM (BiLSTM) network. The proposals achieve high performance score while converging quickly, reducing training time and allowing room for future improvements.

The SLP achieved a test accuracy of 0.9682, slightly surpassing the BiLSTM model's score of 0.9674.

## Dataset
The dataset is composed by time-frequency spectrogram images of gravitational waves affected by random noise, causing the presence of glitches. Each gravitational wave is captured by four images with different time windows.

It includes 7881 instances with label from 22 classes.

The dataset is split in:
- 6093 instances for training;
- 667 instances for validation;
- 1121 instances for test.

### Download Link
https://mega.nz/file/RAY2xDCC#Ky0WeV72GsHujEsWQ_w6AsS20-kKPFU9yNW9dLAdzA0

# Repository Structure
- `/models` is the folder containing the CNNs, BiLSTM and SLP models after training.
- `/dataset` is the folder containing the initial dataset partitioned in four parts, each associated with glitch images with a specific time window, and the four datasets of feature vectors. 

# Execution steps for model training and validation 
In order to perform the training of all the CNNs and of the BiLSTM and SLP, the execution must follow these steps:

1. Run the `branch_gravity.m` program to train the four CNNs, one for each glitch time window;
2. Once the CNNs are saved in `/models`, run `dataset_create.m` to create the four datasets of feature vectors, each related to glitches of a specific time window;
3. Run the `bilstm_gravity.m` and `slp_gravity.m` to train respectively the BiLSTM and SLP models;
4. Run `met_metrics.m` to test the performance of either or both the BiLSTM and SLP models, with also the plot of the confusion matrix. 

# Authors
- Federico Pivotto, 2121720
- Fabrizio Genilotti, 2119281
- Leonardo Egidati, 2106370
