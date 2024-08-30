# Deep Learning for Glitch Classification in Gravitational Waves: a Comparison of CNN-BiLSTM and CNN-SLP Models
In many fields of physics the gravitational waves are studied to deepen the understanding of the fundamentals of physics and of many astronomic processes. The detection of gravitational waves can be achieved by laser-interferometric detectors, which are sensitive to changes in distance under the scale of atomic nuclei. Although the instruments are isolated from non-astrophysical noise, the detectors are still susceptible to instrumental and environmental noise, which can cause gravitational waves to be affected by phenomena called "glitches" that hinder research.

Building on the Gravity Spy project, which aims to classify glitches affecting gravitational wave detectors into morphological families, this paper proposes a deep learning technique to make a contribution to the accurate classification of these glitch phenomena.
In this work, the Gravity Spy dataset is presented and used for the classification task. The dataset consists of time-frequency images of glitches, organized in 22 classes based on glitch morphology. In particular, for each gravitational wave, the dataset includes a set of four images captured with four different time windows.

In this work we compare two deep learning network designs for glitch classification.
Both approaches consists in an initial set of three CNNs, each processing images associated to a specific time window to extract feature vectors. On the second part, which performs the classification task, the first approach uses a Single-layer Perceptron (SLP) network, while the second approach uses a Bidirectional LSTM (BiLSTM) network. The proposals achieve high performance score while converging quickly, reducing training time and allowing room for future improvements.

# Gravity Glitch Network
Deep neural network designed for the classification of glitches in spectrogram images of gravitational waves. 

Each gravitational wave is associated with a set of four time-frequency images, each differing in time window length. The images with the shortest time window were discarded due to the poor performance in test with trained CNN with respect to other time windows.

The network consists of two main parts: the first part employs three parallel CNNs, each processing the same time-frequency image captured with different time windows. The second part, responsible for the classification task, follows two possible approaches: a BiLSTM to capture dependencies between features from different time-windows, or a Single Layer Perceptron (SLP) to capture dependencies by aggregating the features of all the time-windows.

The SLP achieved a test accuracy of 0.9691, slightly surpassing the BiLSTM model's score of 0.9674.

## Dataset
The dataset is composed by time-frequency spectrogram images of gravitational waves affected by random noise, causing the presence of glitches. Each gravitational wave is captured by four images with different time windows.

It includes 7881 instances with label from 22 classes.

The dataset is split in:
- 6093 instances for training;
- 667 instances for validation;
- 1121 instances for test.

### Download Link
https://mega.nz/file/RAY2xDCC#Ky0WeV72GsHujEsWQ_w6AsS20-kKPFU9yNW9dLAdzA0

# Authors
- Federico Pivotto, 2121720
- Fabrizio Genilotti, 2119281
- Leonardo Egidati, 2106370