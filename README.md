# Gravity glitch network
Deep neural network designed for the classification of glitches in spectrogram images of gravitational waves. 

Each gravitational wave is associated with a set of four time-frequency images, each differing in time window length. The images with the shortest time window were discarded due to the poor performance in test with trained CNN w.r.t. other time windows.

The network consists of two main parts: the first part employs three parallel CNNs, each processing the same time-frequency image captured with different time windows. The second part, responsible for the classification task, follows two possible approaches: a BiLSTM to capture dependencies between features from different time-windows, or a fully connected layer to capture dependencies by aggregating the features of all the time-windows.

Both network architectures acheived the same test accuracy.

## Dataset
The dataset is composed by time-frquency spectrogram images of gravitational waves affected by random noise, causing the presence of glitches. Each gravitational wave is captured by four images with different time windows.

It includes 7881 instances with label from 22 classes.

The dataset is split in:
- 6093 instances for training;
- 667 instances for validation;
- 1121 instances for test.

### Download link
https://mega.nz/file/RAY2xDCC#Ky0WeV72GsHujEsWQ_w6AsS20-kKPFU9yNW9dLAdzA0

### Overleaf paper link
https://www.overleaf.com/4625382329nxbkqbskjqyw#866409
