# Group-wise Scaling and Orthogonal Decomposition for Domain-Invariant Feature Extraction in Face Anti-Spoofing. (ICCV 2025)
This repository contains the official implementation of the ICCV 2025 paper.
[[Arxiv](https://arxiv.org/abs/2507.04006)] [[Project](https://seungjinjung.github.io/project/GD-FAS.html)]

Authors: Seungjin Jung, Kanghee Lee, Younghyung Jeong, Haeun Noh, Jungmin Lee, and Jongwon Choi*

# Information on the Datasets Used in the paper
The experiments in this paper are conducted on several publicly available datasets widely used in face anti-spoofing research:

| Dataset | Paper | Datalink |
|---|---|---|
|CASIA-MFSD          | [A face antispoofing database with diverse attacks](https://ieeexplore.ieee.org/document/6199754) <!--https://mega.nz/file/9BFDiKqT#VLexsuFDZjoA97c1J_h9hInm8AG75h6kG-TUfm3hYwg-->||
|OULU-NPU            | [OULU-NPU: A Mobile Face Presentation Attack Database with Real-World Variations](https://ieeexplore.ieee.org/document/7961798) | [Data](https://sites.google.com/site/oulunpudatabase/)|
|Idaip Replay-Attack | [On the effectiveness of local binary patterns in face anti-spoofing](https://ieeexplore.ieee.org/document/6313548)             | [Data](https://www.idiap.ch/en/scientific-research/data/replayattack) |
|MSU-MFSD            | [Face Spoof Detection With Image Distortion Analysis](https://ieeexplore.ieee.org/document/7031384)                             | [Data](https://drive.google.com/drive/folders/1nJCPdJ7R67xOiklF1omkfz4yHeJwhQsz) |
|CelebA-Spoof        | [CelebA-Spoof: Large-Scale Face Anti-Spoofing Dataset with Rich Annotations](https://arxiv.org/abs/2007.12342)                  | [Data](https://drive.google.com/drive/folders/1OW_1bawO79pRqdVEVmBzp8HSxdSwln_Z) |
|CASIA-Surf          | [CASIA-SURF: A Large-scale Multi-modal Benchmark for Face Anti-spoofing](https://arxiv.org/abs/1908.10654)             | [Data](https://sites.google.com/view/face-anti-spoofing-challenge/dataset-download/casia-surfcvpr2019)|
|CASIA-CeFA          | [CASIA-SURF CeFA: A Benchmark for Multi-modal Cross-ethnicity Face Anti-spoofing](https://arxiv.org/abs/2003.05136)             | [Data](https://sites.google.com/view/face-anti-spoofing-challenge/dataset-download/casia-surf-cefacvpr2020) |
|WCMA                | [Biometric Face Presentation Attack Detection with Multi-Channel Convolutional Neural Network](https://arxiv.org/abs/1909.08848)             | [Data](https://www.idiap.ch/en/scientific-research/data/wmca) |

# How to Structure Data Directories
* {dataset}&emsp;: name of dataset<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;e.g.) CASIA, OULU, Idaip, MSU, CeFA, Surf, WMCA, CelebA
* {video}&emsp;&emsp;: Name of the directory in the format {session}_{video_name} (without file extension)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;e.g.) Idiap - [fixed_attack_highdef_client003_session01_highdef_photo_adverse,<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;hand_attack_highdef_client003_session01_highdef_photo_adverse, ...]
* {img}&emsp;&emsp;&nbsp;&nbsp;&nbsp;: name of image as number of frame (every 5 frames)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;e.g.) CASIA - [005.png, 010.png, ...]

```text
dataset
|-- {dataset}
|   |-- train
|   |   |-- attack
|   |   |   |-- {video}
|   |   |       |-- {img}
|   |   |-- live
|   |   |   |-- {video}
|   |   |       |-- {img}
|   |-- test
|   |   |-- attack
|   |   |   |-- {video}
|   |   |       |-- {img}
|   |   |-- live
|   |   |   |-- {video}
|   |   |       |-- {img}
```

# How to Preprocess Data
1. Frame extraction. Extract every 5th frame from each video. ( Following [[SAFAS](https://openaccess.thecvf.com/content/CVPR2023/papers/Sun_Rethinking_Domain_Generalization_for_Face_Anti-Spoofing_Separability_and_Alignment_CVPR_2023_paper.pdf)])
2. Face processing. Detect and align faces using [MTCNN](https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection). We use a padding ratio of 0.6.


# How to Train
python `GD-FAS.py --gs --protocol O_C_I_to_M`

# Information about Previous Researches

| Dataset | Conference | Paper | Github |
|---|---|---|---|
|SAFAS|CVPR 23|[Rethinking domain generalization for face anti-spoofing: Separability and alignment](https://openaccess.thecvf.com/content/CVPR2023/papers/Sun_Rethinking_Domain_Generalization_for_Face_Anti-Spoofing_Separability_and_Alignment_CVPR_2023_paper.pdf)|[Github](https://github.com/sunyiyou/SAFAS)|
|FLIP|ICCV 23|[FLIP: Cross-domain Face Anti-spoofing with Language Guidance](https://openaccess.thecvf.com/content/ICCV2023/papers/Srivatsan_FLIP_Cross-domain_Face_Anti-spoofing_with_Language_Guidance_ICCV_2023_paper.pdf)|[Github](https://github.com/koushiksrivats/FLIP)|
|CFPL|CVPR 24|[CFPL-FAS: Class Free Prompt Learning for Generalizable Face Anti-spoofing](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_CFPL-FAS_Class_Free_Prompt_Learning_for_Generalizable_Face_Anti-spoofing_CVPR_2024_paper.pdf)|-|
|BUDoPT|ECCV 24|[Bottom-up domain prompt tuning for generalized face anti-spoofing](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08866.pdf)|-|
|TF-FAS|ECCV 24|[TF-FAS: Twofold-Element Fine-Grained Semantic Guidance for Generalizable Face Anti-Spoofing](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01098.pdf)|[Github](https://github.com/xudongww/TF-FAS)|

# Citation
If you use this work in your research or applications, please cite:

```bibtex
@article{jung2025group,
  title={Group-wise Scaling and Orthogonal Decomposition for Domain-Invariant Feature Extraction in Face Anti-Spoofing},
  author={Jung, Seungjin and Lee, Kanghee and Jeong, Yonghyun and Noh, Haeun and Lee, Jungmin and Choi, Jongwon},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  year={2025}
}
```



