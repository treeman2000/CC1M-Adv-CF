# CC1M-adv-C/F: Two million-scale datasets for adversarial robustness evaluation of vision models (classifiers and feature extractors)

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
| Swin-L | A comprehensive study on robustness of image classification models: Benchmarking and rethinking |
| ConvNeXt-L | A comprehensive study on robustness of image classification models: Benchmarking and rethinking |
| ViT-B + ConvStem | Revisiting adversarial586 training for imagenet: Architectures, training and generalization across threat models |
| RaWideResNet-101-2 | Robust principles: Architectural design principles for adversarially robust cnns |


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
| vgg16 | Very deep convolutional networks for large-scale image recognition|
| resnet101 | Deep residual learning for image recognition|
| efficient net | Efficientnet: Rethinking model scaling for convolutional neural networks|
| convnext_base | A convnet for the 2020s|
| vit_base_patch16_224 | An image is worth 16x16 words: Transformers for image recognition at scale|
| vit_base_patch16_224.dino | Emerging properties in self-supervised vision transformers|
| beit_base_patch16_224 |Beit: Bert pre-training of image transformers|
| swin_base_patch4_window7_224 | Swin transformer: Hierarchical vision transformer using shifted windows|


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

## Acknowledgements
Our work is based on the CC3M dataset.
website: https://ai.google.com/research/ConceptualCaptions/
paper: https://aclanthology.org/P18-1238/
github: https://github.com/google-research-datasets/conceptual-captions

## Cite Us
If you find our datasets interesting and helpful, please consider citing us in your research or publications:
