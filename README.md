# Generative Models for Low Rank Representation and Reconstruction of Videos
This repository provides implementation of the algorithm of the paper:

In this paper, we propose a generative model to learn compact latent codes that can efficiently represent and reconstruct a video sequence from its missing or under-sampled measurements. We propose a low rank constraint on the corresponding latent codes of the neighboring frames in the video sequence which allow us to represent the whole video sequence with very few number of latent codes. We also could linearize the articulation manifold of a video sequence by imposing low-rank structure on the latent codes. Furthermore, we demonstrate that even if the video sequence does not belong to the range of a pretrained network, we can still recover the true video sequence by jointly updating the latent codes and the weights of the generative network.

Some results:

|Reconstructions (Joint optimization with rank=2 (linear) constraint)|Corresponding latent code representation|
| --- | --- |
|![rot_mnist](https://user-images.githubusercontent.com/32584505/52319847-a4444100-2980-11e9-8151-087a2ef22018.png)| ![z_manifold_rot_mnist](https://user-images.githubusercontent.com/32584505/52319883-cfc72b80-2980-11e9-8cd6-1bbac809de70.png)|
|![walking](https://user-images.githubusercontent.com/32584505/52320027-95aa5980-2981-11e9-8127-29e30e857b25.png)| ![z_manifold_walking](https://user-images.githubusercontent.com/32584505/52320037-a5c23900-2981-11e9-9607-9c803ebed7d1.png)|




# Requirements
1. python 2.7 (Anaconda for python 2.7: https://www.anaconda.com/distribution/)
2. pytorch 0.4.1 (To install pytorch in Anaconda: conda install pytorch torchvision cudatoolkit=9.0 -c pytorch)
3. matplotlib 2.2.3 (Installing anaconda will automatically install it.)
4. scipy 2.2.3 (Installing anaconda will automatically install it.)
5. numpy 1.15.1 (Installing anaconda will automatically install it.)

The code is written for gpu enabled devices. You need to have nvidia driver installed to run it. (To install nvidia driver in Ubuntu OS: "sudo apt-get install nvidia-384" or "sudo apt-get install nvidia-current")

# Citation
If we use this code in your research, please cite this paper:





