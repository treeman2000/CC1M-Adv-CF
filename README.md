# CC1M-adv-C/F: Datasets for robustness evaluation

Robustness evaluation is particularly important in the field of AI, especially when deep learning and machine learning models are widely used in real-world tasks. The challenge in this field is that many models may show fragility when facing various uncertainties and perturbations. Robustness evaluation aims to quantify and improve the stability and reliability of the model in different scenarios, so as to ensure that it can cope with unforeseen input changes, adversarial attacks or data bias.

Considering the importance of robustness, we need a method to conveniently evaluate the robustness of the model. To this end, we selected one million images from the CC3M dataset to form the initial CC1M dataset. Based on this, we constructed two robustness evaluation datasets, which were generated based on different methods, one based on the classification head, called CC1M-adv-C, and the other based on the feature, called CC1M-adv-F. We provide supporting code to facilitate testing of the model.

<p align="center">
<img src="./cc1m.jpg"  width="480px" height="290px" alt="CC1M-adv" title="CC1M-adv" align="center"></img>
</p>

## CC1M-adv-C
### Dataset Description
We generate highly transferable adversarial examples by perturbing inputs in a way that affects multiple classification models simultaneously.
We selected 4 mainstream adversarially trained models from the RobustBench library for generating adversarial examples, which include various architectures and defence methods, as detailed below:
| model name | paper |
| --- | --- |
| Swin-L | C. Liu, Y. Dong, W. Xiang, X. Yang, H. Su, J. Zhu, Y. Chen, Y. He, H. Xue, and S. Zheng, “A comprehensive study on robustness of image classification models: Benchmarking and rethinking,” arXiv preprint arXiv:2302.14301, 2023. |
| ConvNeXt-L | C. Liu, Y. Dong, W. Xiang, X. Yang, H. Su, J. Zhu, Y. Chen, Y. He, H. Xue, and S. Zheng, “A comprehensive study on robustness of image classification models: Benchmarking and rethinking,” arXiv preprint arXiv:2302.14301, 2023. |
| ViT-B + ConvStem | N. D. Singh, F. Croce, and M. Hein, “Revisiting adversarial586 training for imagenet: Architectures, training and generalization across threat models,” in NeurIPS, 2024 |
| RaWideResNet-101-2 | S. Peng, W. Xu, C. Cornelius, M. Hull, K. Li, R. Duggal, M. Phute, J. Martin, and D. H. Chau, “Robust principles: Architectural design principles for adversarially robust cnns, ”arXiv preprint arXiv:2308.16258, 2023. |


### File Structure

```
cc1m
    |--000000000.jpg
    |--000000001.jpg
    |--000000002.jpg
    ...
```

### Download
https://huggingface.co/datasets/xingjunm/CC1M-Adv-C

### Usage
This dataset can be used with the code we provide to test the relative robustness of the model.
Our robustness evaluation code can be found at: https://github.com/OpenTAI/taiadv/blob/main/taiadv/vision/white-box


## CC1M-adv-F
### Dataset Description
We generate highly transferable adversarial examples by perturbing in parallel at the feature layer using multiple pre-trained image encoders.
We selected 8 mainstream feature extractors from the timm library for generating adversarial examples, which include various model architectures and pre-training methods, as detailed below:
| model name | paper |
| --- | --- |
| vgg16 | Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition[J]. arXiv preprint arXiv:1409.1556, 2014. |
| resnet101 | He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778. |
| efficient net | Tan M, Le Q. Efficientnet: Rethinking model scaling for convolutional neural networks[C]//International conference on machine learning. PMLR, 2019: 6105-6114.
| convnext_base | Liu Z, Mao H, Wu C Y, et al. A convnet for the 2020s[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022: 11976-11986. |
| vit_base_patch16_224 | Dosovitskiy A, Beyer L, Kolesnikov A, et al. An image is worth 16x16 words: Transformers for image recognition at scale[J]. arXiv preprint arXiv:2010.11929, 2020. |
| vit_base_patch16_224.dino | Caron M, Touvron H, Misra I, et al. Emerging properties in self-supervised vision transformers[C]//Proceedings of the IEEE/CVF international conference on computer vision. 2021: 9650-9660. |
| beit_base_patch16_224 | Bao H, Dong L, Piao S, et al. Beit: Bert pre-training of image transformers[J]. arXiv preprint arXiv:2106.08254, 2021. |
| swin_base_patch4_window7_224 | Liu Z, Lin Y, Cao Y, et al. Swin transformer: Hierarchical vision transformer using shifted windows[C]//Proceedings of the IEEE/CVF international conference on computer vision. 2021: 10012-10022. |


### File Structure

```
cc1m
    |--000000000.jpg
    |--000000001.jpg
    |--000000002.jpg
    ...
```

### Download
https://huggingface.co/datasets/xingjunm/CC1M-Adv-F

### Usage
This dataset can be used with the code we provide to test the relative robustness of the model.
Our robustness evaluation code can be found at: https://github.com/OpenTAI/taiadv/blob/main/taiadv/vision/black-box
