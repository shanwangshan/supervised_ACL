# Supervised_ACL

This git repo presents the pytorch-based implementation of the supervised angular contrastive loss from paper [*"Self-supervised learning of audio representations using angular contrastive loss"*](https://arxiv.org/abs/2211.05442). The motivation of applying ACL is demonstrated [here](https://github.com/shanwangshan/problems_of_infonce), the Self-supervised ACL is implementated [here](https://github.com/shanwangshan/Self_supervised_ACL), and the feature quality analysis is presented [here](https://github.com/shanwangshan/uniformity_tolerance).


To create the features and save them in *.hdf5* format, run the command below,

`` python create_data.py -p config/params.yaml ``

Note that, for different subset of the FSDnoisy18K, change the setting of *train_data*(all, clean, noisy, noisy\_small) from config/params.yaml file.

To run the training script for different alpha values, here we submit an array job shown as below,

`` sbatch gpu_sbatch.sh ``

The traning loss is angular contrastive loss (ACL) shown as below,

$ACL = \alpha * L1 + (1-\alpha) * L2$, where L1 is cross entropy loss(CEL) and L2 is angular margin loss.

To test, e.g., clean data type and $\alpha$ is 5, run the command below,

`` python test.py -model_type clean -alpha 5 ``



The results of different data type and different alpha values are shown in the tabels below,

| alpha | clean       | all         | noisy       | noisy\_small |
|-------|-------------|-------------|-------------|--------------|
| 0.1   | 0.56        | 0.674       | 0.649       | 0.345        |
| 0.2   | 0.614       | 0.698       | 0.654       | ***0.399***  |
| 0.3   | 0.61        | 0.719       | ***0.683*** | 0.393        |
| 0.4   | 0.621       | 0.727       | ***0.683*** | 0.335        |
| 0.5   | ***0.645*** | 0.712       | 0.677       | 0.39         |
| 0.6   | 0.63        | 0.713       | 0.669       | 0.333        |
| 0.7   | 0.579       | ***0.736*** | 0.675       | 0.35         |
| 0.8   | 0.581       | 0.727       | 0.671       | 0.316        |
| 0.9   | 0.586       | 0.716       | 0.661       | 0.331        |
| *1.0* | *0.602*     | *0.701*     | *0.664*     | *0.337*      |


# Acknowledgement
This code is adapted from [here](https://github.com/edufonseca/icassp19) by  Fonseca et al.
