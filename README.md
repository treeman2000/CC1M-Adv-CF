# CC1M-adv-C/F: Two million-scale datasets for adversarial robustness evaluation of vision models (classifiers and feature extractors)

Current evaluations of adversarial robustness for vision models are mostly small-scale, often based on subsets of CIFAR-10 or ImageNet. We believe that large-scale (million-scale) assessments are crucial for advancing the field. To facilitate large-scale adversarial robustness testing for vision models, we have constructed a dataset called CC1M based on CC3M, by removing outlier images (based on the LID metric) and sampling one million images. Subsequently, based on CC1M, we have created CC1M-Adv-C and CC1M-Adv-F using the following methods:
- Probability Margin Attack (PMA), which introduces a probability margin loss to boost attack effectiveness of individual attacks.
- 

<p align="center">
<img src="./cc1m.jpg"  width="480px" height="290px" alt="CC1M-adv" title="CC1M-adv" align="center"></img>
</p>

## CC1M-adv-C
### Dataset Description
We focus on image classification models and propose a novel individual attack method, Probability Margin Attack (PMA), which defines the adversarial margin in the probability space rather than the logits space. We generate highly transferable adversarial examples by perturbing inputs in PMA, thereby affecting multiple classification models simultaneously.  
We selected 4 mainstream adversarially trained models from the RobustBench library for generating adversarial examples, which include various architectures and defence methods. The following table shows the models we used.
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
We selected 8 mainstream feature extractors from the timm library for generating adversarial examples, which include various model architectures and pre-training methods. The following table shows the models we used.
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
