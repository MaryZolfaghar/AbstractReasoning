Notes: 
1) Between the code cleanup, updating to the latest PyTorch version and updating to the latest Nvidia APEX version the training curve changed from what is shown in the paper and the training accuracy now shows more overfitting. However, the validation accuracy curve and the final test accuracies still match the results reported in the paper.
2) The model was trained with Nvidia's mixed precision library AMP. The results can be reproduced by using AMP without its cpp and cuda extensions (see APEX package installation instructions) since using the extensions can lead to "illegal memory access" exceptions from cuda.
3) The reported hyperparameters are selected for a large batch size which might not fit a single gpu's memory alone, so training on less than 4 gpus will most likely require changing the batch size and hyperparameters.

Tested using: 
Python 3.6
PyTorch 1.3.1
apex 0.1 (only for mixed precision training)
tensorflow 1.13.1 (only for tensorboard logging)

Instructions:
1) extract the PGM dataset and keep the neutral split's samples in a subfolder named "neutral"
2) preprocess the dataset using "python preprocess.py path/to/dataset/parent/folder"
3) train the model using "python train_mixed_precision_distributed.py path/to/dataset/parent/folder"
4) evaluate the model using "python test.py path/to/dataset/parent/folder"
