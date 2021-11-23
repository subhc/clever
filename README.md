
## The Curious Layperson: Fine-Grained Image Recognition without Expert Labels 
#### Subhabrata Choudhury, Iro Laina, Christian Rupprecht, Andrea Vedaldi
### [![ProjectPage](https://img.shields.io/badge/-Project%20Page-magenta.svg?style=for-the-badge&color=white&labelColor=magenta)](https://www.robots.ox.ac.uk/~vgg/research/clever/) [![Conference](https://img.shields.io/badge/BMVC%20Oral-2021-purple.svg?style=for-the-badge&color=f1e3ff&labelColor=purple)](https://www.bmvc2021-virtualconference.com/conference/papers/paper_0229.html)    [![arXiv](https://img.shields.io/badge/arXiv-2111.03651-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2111.03651)

### Training:
Training is performed in two stages. Code and training script for each stage of training code has been put in the corresponding folder. Please use the environment file to first set up the conda environment.

### Evaluation:
You can find evaluation code in the [evaluation notebook](evaluate.ipynb).

### Pretrained weights and dataset files:

| Description | Size | Link |
| ---- | -----| ----|
| Dataset files and precomputed features | 2.1GB | [here](https://www.robots.ox.ac.uk/~vgg/research/clever/downloads/clever_datasets_231121.zip) |
| Pretrained weights | 4.9GB| [here](https://www.robots.ox.ac.uk/~vgg/research/clever/downloads/clever_checkpoints_231121.zip) |

Please unzip them at the root, where this README file is.


### Abstract:
<sup> Most of us are not experts in specific fields, such as ornithology. Nonetheless, we do have general image and language understanding capabilities that we use to match what we see to expert resources. This allows us to expand our knowledge and perform novel tasks without ad-hoc external supervision. On the contrary, machines have a much harder time consulting expert-curated knowledge bases unless trained specifically with that knowledge in mind. Thus, in this paper we consider a new problem: fine-grained image recognition without expert annotations, which we address by leveraging the vast knowledge available in web encyclopedias. First, we learn a model to describe the visual appearance of objects using non-expert image descriptions. We then train a fine- grained textual similarity model that matches image descriptions with documents on a sentence-level basis. We evaluate the method on two datasets and compare with several strong baselines and the state of the art in cross-modal retrieval. </sup>


### Citation   
```
@inproceedings{choudhury2021curious,
author = {Choudhury, Subhabrata and Laina, Iro and Rupprecht, Christian and Vedaldi, Andrea},
booktitle = {British Machine Vision Conference}
title = {The Curious Layperson: Fine-Grained Image Recognition without Expert Labels}
volume = {32},
year = {2021}
}
```   
