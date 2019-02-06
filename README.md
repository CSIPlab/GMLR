# GMLR
Generative Models for Low Rank Representation and Reconstruction of Videos

# Abstract
Finding compact representation of videos is an essential component in almost every problem related to video processing or understanding. In this paper, we propose a generative model to learn compact latent codes that can efficiently represent and reconstruct a video sequence from its missing or under-sampled measurements. We assume we have a generative network that was trained to map a compact code into an image. We first demonstrate that if a video sequence belongs to the range of the pretrained generative network, then we can recover it by estimating the underlying compact latent codes. Then we demonstrate that even if the video sequence does not belong to the range of a pretrained network, we can still recover the true video sequence by jointly updating the latent codes and the weights of the generative network. 
To avoid overfitting in our model, we regularize the recovery problem by imposing low-rank and similarity constraints on the latent codes of the neighboring frames in the video sequence. We demonstrate the performance of our methods on a variety of video sequences. 

# Requirements
1. python 2.7 (Anaconda for python 2.7: https://www.anaconda.com/distribution/)
2. pytorch 0.4.1 (To install previous versions of pytorch: https://pytorch.org/get-started/previous-versions/)
3. matplotlib 2.2.3 (Installing anaconda will automatically install it.)
4. scipy 2.2.3 (Installing anaconda will automatically install it.)
5. numpy 1.15.1 (Installing anaconda will automatically install it.)

The code is written for gpu enabled devices. You need to have nvidia driver installed to run it.
