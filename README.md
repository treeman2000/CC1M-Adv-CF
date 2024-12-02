# CC1M-Adv-C/F: Million-Scale Datasets for Adversarial Robustness Evaluation of Vision Models
  
<p align="center">
<img src="./cc1m.jpg"  width="720" height="435px" alt="CC1M-Adv" title="CC1M-Adv" align="center"></img>
</p>

## 📌 Dataset Description

Current evaluations of adversarial robustness in vision models are often limited in scale, typically relying on subsets of datasets like CIFAR-10 or ImageNet. We believe that large-scale assessments, on the order of millions of samples, are essential for advancing the field. To enable large-scale testing of adversarial robustness in vision models, we have constructed a new dataset, **CC1M**, based on [**CC3M**](https://github.com/google-research-datasets/conceptual-captions), by removing outlier images and keeping only one million images.

Building on **CC1M**, we have created two adversarial variants: **CC1M-Adv-C** and **CC1M-Adv-F**, which are designed for different types of vision models.

- **CC1M-Adv-C** is intended for evaluating **image classification models**. It was created by **Probability Margin Attack (PMA)** [1], a novel attack method we propose for image classification models. We also use a **Surrogate Ensemble** technique to boost the transferability of the adversarial noise to different models.
   
- **CC1M-Adv-F** is designed for testing **non-classification models**. It was created by a **cosine similarity based feature attack** [2]. We also use the **Surrogate Ensemble** technique to boost the transferability across datasets and models.


Demo evaluations can be found on our **Vision Safety Platform**: [https://github.com/OpenTAI/taiadv/blob/main/taiadv/vision](https://github.com/OpenTAI/taiadv/blob/main/taiadv/vision).


## 🐥 CC1M
### 📌 Dataset Description
**CC1M** is a large-scale evaluation dataset containing one million images selected from **CC3M**, which consists of image-caption pairs that cover a wide range of objects, scenes, and visual concepts. To construct **CC1M**, we first removed unavailable images, such as those displaying a “this image is unavailable” message due to expired URLs. Next, we filtered out noisy images that lack meaningful semantic content, such as random icons. To further refine the dataset, we applied the **Local Intrinsic Dimensionality (LID)** metric, which is effective for detecting adversarial images, backdoor images, and low-quality data that could negatively impact self-supervised contrastive learning. By calculating LID scores based on **CLIP** embeddings, we identified outliers using the **Median Absolute Deviation (MAD)** method and retained only those images whose LID scores were close to the median. This approach ensures that **CC1M** consists of high-quality, diverse images, making it well-suited for robust evaluation tasks.

### 📂 File Structure
```
cc1m
    |--000000000.jpg
    |--000000001.jpg
    |--000000002.jpg
    ...
```


## 🐝 CC1M-Adv-C
### 📌 Dataset Description
**CC1M-Adv-C** is created to evaluate **image classification models**, by our new attack **Probability Margin Attack (PMA)**, which defines the adversarial margin in the probability space rather than the logits space. Using **PMA**, we generate highly transferable adversarial examples, referred to as **CC1M-Adv-C**, by generating perturbations on an ensemble of 4 surrogate models simultaneously. 

To create these adversarial examples, we utilized four surrogate models from the [**RobustBench**](https://robustbench.github.io/) leaderboard, which include a range of architectures and defense strategies. The table below provides an overview of the surrogate models used.

| Surrogate Model | Paper |
| --- | --- |
| Swin-L | A comprehensive study on robustness of image classification models: Benchmarking and rethinking |
| ConvNeXt-L | A comprehensive study on robustness of image classification models: Benchmarking and rethinking |
| ViT-B + ConvStem | Revisiting adversarial586 training for imagenet: Architectures, training and generalization across threat models |
| RaWideResNet-101-2 | Robust principles: Architectural design principles for adversarially robust cnns |


### 📂 File Structure

```
cc1m_adv_c
    |--000000000.jpg
    |--000000001.jpg
    |--000000002.jpg
    ...
```

### ⬇️ Download
[Download CC1M-Adc-C dataset](https://huggingface.co/datasets/xingjunm/CC1M-Adv-C)

### 🌲 Usage
Our provide [example code](https://github.com/OpenTAI/taiadv/blob/main/taiadv/vision/white-box) for using the dataset.

<br>
<br>

## 🐞 CC1M-Adv-F
### 📌 Dataset Description
**CC1M-Adv-F** is created to evalute vision feature extractors. It was generated based on pre-trained ViTs on ImageNet by our **Downstream Transfer Attack (DTA)**, which identifies the most vulnerable and transferable layer of pre-trained (by MAE or SimCLR) ViTs. We improves the transferability of the generated adversarial examples by perturbing the feature layer in parallel using multiple pre-trained image encoders. For this, we selected eight widely used feature extractors from the [timm library](https://huggingface.co/docs/timm/en/index), which encompass a variety of model architectures and pre-training methods. The table below provides an overview of the surrogate models we used.

| Surrogate Model | Paper |
| --- | --- |
| vgg16 | Very deep convolutional networks for large-scale image recognition|
| resnet101 | Deep residual learning for image recognition|
| efficient net | Efficientnet: Rethinking model scaling for convolutional neural networks|
| convnext_base | A convnet for the 2020s|
| vit_base_patch16_224 | An image is worth 16x16 words: Transformers for image recognition at scale|
| vit_base_patch16_224.dino | Emerging properties in self-supervised vision transformers|
| beit_base_patch16_224 |Beit: Bert pre-training of image transformers|
| swin_base_patch4_window7_224 | Swin transformer: Hierarchical vision transformer using shifted windows|


### 📂 File Structure

```
cc1m_adv_f
    |--000000000.jpg
    |--000000001.jpg
    |--000000002.jpg
    ...
```

### ⬇️ Download
[Download CC1M-Adc-F dataset](https://huggingface.co/datasets/xingjunm/CC1M-Adv-F)

### 🌲 Usage
We provide [example code](https://github.com/OpenTAI/taiadv/blob/main/taiadv/vision/black-box) for using the dataset.


## ⭐ Acknowledgements
Our work is created based on the CC3M dataset.
Website: https://ai.google.com/research/ConceptualCaptions/
Paper: https://aclanthology.org/P18-1238/
GitHub: https://github.com/google-research-datasets/conceptual-captions


## 📜 Cite Us
If you use this dataset in your research, please cite it as follows:

```
[1]
@article{xie2024towards,
  title={Towards Million-Scale Adversarial Robustness Evaluation With Stronger Individual Attacks},
  author={Xie, Yong and Zheng, Weijie and Huang, Hanxun and Ye, Guangnan and Ma, Xingjun},
  journal={arXiv:2411.15210},
  year={2024}
}

[2]
@article{zheng2024downstream,
  title={Downstream Transfer Attack: Adversarial Attacks on Downstream Models with Pre-trained Vision Transformers},
  author={Zheng, Weijie and Ma, Xingjun and Huang, Hanxun and Wu, Zuxuan and Jiang, Yu-Gang},
  journal={arXiv:2408.01705},
  year={2024}
}
```
