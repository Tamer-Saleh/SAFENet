<h1>SAFE-Net: Flood mapping from SAR imagery with a shift-attention and frequency-gated vision transformer</h1>

<div>
    <h4 align="center">
    <a href='https://scholar.google.com/citations?hl=en&user=KAmm5ZkAAAAJ&view_op=list_works&sortby=pubdate' target='_blank'>Tamer Saleh</a><sup>1,2</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=SAUCVsEAAAAJ&hl=en' target='_blank'>Gui-Song Xia</a><sup>✉1</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=-aVL-UQAAAAJ&hl=en' target='_blank'>Wen Yang</a><sup>✉1</sup>&emsp;
    <a href='' target='_blank'>Mohamed Ihmeida</a><sup>3</sup>&emsp;
    <a href=''>Zhuohong Li</a><sup>4</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=WKKVqDgAAAAJ&hl=en' target='_blank'>Shimaa Holail</a><sup>1</sup>&emsp;
    </h4>
</div>

<div>
    <h4 align="center">
    <sup>1</sup>Wuhan University&emsp;
    <sup>2</sup>Benha University&emsp;
    <sup>3</sup>Birmingham City University&emsp;
    <sup>4</sup>Duke University&emsp;
    </h4>

</div>


## Requirements

Before using this repository, make sure you have the following prerequisites installed:

- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)

> torch==2.6.0
  torchaudio==2.6.0
  torchvision==0.21.0

Tested using Python 3.9.0 on a 64-bit Windows operating system.
```bash
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
```

### Installation

To get started, create the [conda](https://docs.conda.io/projects/conda/en/stable/) environment named `SAFE-FD` by executing the following command:
```bash
conda create -n SAFE_FD python=3.9.0 -y
```

Then activate the environment:
```bash
conda activate SAFE_FD
```

## Supported Datasets

Download the datasets and place them in the `Benchmark` folder.

Dataset | Name | Link
:-:|:-:|:-:
Ombria-Net | `OmbriaS1` | [source](https://ieeexplore.ieee.org/document/9723593)
DAM-Net | `S1GFloods` | [source](https://www.sciencedirect.com/science/article/pii/S0924271624002168)
New-Dataset | `S1GFloods+7` | [baidu drive](https://pan.baidu.com/s/1SkFGxfFIf7nDIIbDI4Zlrg?pwd=i8tm) Passward: (i8tm)


### :point_right: Data Structure

```yaml
For S1GFloods+7 dataset, please respect the following structure: 
train/
     A/                        Images of Time 1 before the flood event
     B/                        Images of Time 2 after the flood event          
     GT/                        Ground truth labels
test/
     A/ 
     B/            
     GT/       
```


## 1. Method
![image-20210228153142126](./results/method.png)

> We propose a shifted attention frequency-gated network (SAFE-Net) for flood detection from bi-temporal SAR image pairs.


### 🔭 Supported SOTA <a name="baselines"></a>

Method | Name | Link
:-:|:-:|:-:
:open_book:	:open_book:	 :open_book: DBF-Net | `DBF-Net` | [paper here](https://ieeexplore.ieee.org/document/10848167)
:open_book:	:open_book:	 :open_book: WBA-Net | `WBA-Net` | [paper here](https://ieeexplore.ieee.org/document/10605827)
:open_book:	:open_book:	 :open_book: GLA-Former | `GLA-Former` | [paper here](https://ieeexplore.ieee.org/document/10766648)
:open_book:	:open_book:	 :open_book: CAS-Net | `CAS-Net` | [paper here ](https://ieeexplore.ieee.org/document/10504920)
:open_book:	:open_book:	 :open_book: PA-Former | `PA-Former` | [paper here](https://ieeexplore.ieee.org/document/9863867)
:open_book:	:open_book:	 :open_book: ViCxLSTM | `ViCxLSTM` | [paper here](https://www.sciencedirect.com/science/article/pii/S1569843225004480)
:open_book:	:open_book:	 :open_book: BiSR-Net | `BiSR-Net` | [paper here](https://ieeexplore.ieee.org/document/9721305)
:open_book:	:open_book:	 :open_book: SCan-Net | `SCan-Net` | [paper here](https://ieeexplore.ieee.org/document/10443352)



## Checkpoints
### OmbriaS1

| Method    |Input size  | F1-score  |       Checkpoints      |
:-:|:-:|:-:|:-:
| DBF-Net   | 256 × 256  | 53.3      | [baidu](https://pan.baidu.com/s/1I29tvP0iYvsX0XEysnoR9g?pwd=jh1i ) |
| WBA-Net   | 256 × 256  | 52.9      | [baidu](https://pan.baidu.com/s/1J2SwF10KSLP80WDmmj3fZw?pwd=rir3) |
| GLA-Former| 256 × 256  | 65.7      | [baidu](https://pan.baidu.com/s/1V3DAbAHTwqXQWNsCTebeHA?pwd=n9a7) |
| CAS-Net   | 256 × 256  | 71.3      | [baidu](https://pan.baidu.com/s/1Z4hKwiHHNP3TraEK3gAYsw?pwd=u3cf) |
| PA-Former | 256 × 256  | 76.7      | [baidu](https://pan.baidu.com/s/1Cg5IceEfMwQXs3wCGECP_g?pwd=gvii) |
| ViCxLSTM  | 256 × 256  | 77.4      | [baidu](https://pan.baidu.com/s/1RF_dVf0_TexdKHO3b9JHEA?pwd=zje9) |
| BiSR-Net  | 256 × 256  | 78.0      | [baidu](https://pan.baidu.com/s/1axnF0VjWGOG7IHqJdspoKg?pwd=9hgc) |
| SCan-Net  | 256 × 256  | 81.3      | [baidu](https://pan.baidu.com/s/1lY-g7_ZJ_i_EU8rd8-2Hag?pwd=t1hr) |
| SAFE-Net  | 256 × 256  | 85.3      | [baidu](https://pan.baidu.com/s/1Lev4P2SCZBgdTA8MEVF73A?pwd=yhtn) |

### S1GFloods

| Method    |Input size  | F1-score  |  Checkpoints   |
:-:|:-:|:-:|:-:
| DBF-Net   | 256 × 256  | 62.4      | [baidu]() |
| WBA-Net   | 256 × 256  | 88.7      | [baidu]() |
| GLA-Former| 256 × 256  | 93.6      | [baidu]() |
| CAS-Net   | 256 × 256  | 86.5      | [baidu]() |
| PA-Former | 256 × 256  | 95.9      | [baidu]() |
| ViCxLSTM  | 256 × 256  | 95.0      | [baidu]() |
| BiSR-Net  | 256 × 256  | 94.2      | [baidu]() |
| SCan-Net  | 256 × 256  | 96.2      | [baidu]() |
| SAFE-Net  | 256 × 256  | 96.8      | [baidu]() |


## Results
### Quantitative
![image-20210228153142126](./results/tables.png)

> Quantitative comparison of flood detection results on the S1GFloods+7 test dataset. The best value for each evaluation metric is highlighted in bold red, while the second-best value is underlined in blue. Each metric is reported as the mean (%) ± standard deviation (per-image average).


### Qualitative
- OmbriaS1

    ![image-20210228153142126](./results/OmbriaS1.png)

    > Qualitative comparison on the OmbriaS1 dataset. From left to right: T1 image, T2 image, Ground Truth, DBF-Net, CAS-Net, WBA-Net, GLA-Former, BiSR-Net, ViCxLSTM, PA-Former, SCan-Net, and the proposed SAFE-Net method (Ours).

- S1GFloods

    ![image-20210228153142126](./results/S1GFloods.png)

    > Qualitative comparison on the S1GFloods dataset. From left to right: T1 image, T2 image, Ground Truth, DBF-Net, CAS-Net, WBA-Net, GLA-Former, BiSR-Net, ViCxLSTM, PA-Former, SCan-Net, and the proposed SAFE-Net method (Ours).

- S1GFloods+7

    ![image-20210228153142126](./results/S1GFloods7.png)

    > Qualitative comparison on the S1GFloods+7 dataset. From left to right: T1 image, T2 image, Ground Truth, DBF-Net, CAS-Net, WBA-Net, BiSR-Net, ViCxLSTM, GLA-Former, SCan-Net, PA-Former, and the proposed SAFE-Net method (Ours).


### :page_with_curl: Citation <a name="citing"></a>

```bibtex
@ARTICLE{tasaleh2026SAFENet,
  Author = {Tamer Saleh, Gui-Song Xia, Wen Yang, Mohamed Ihmeida, Zhuohong Li, Shimaa Holail},
  Title = {SAFE-Net: Flood mapping from SAR imagery with a shift-attention and frequency-gated vision transformer},
  Journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  Year = {2026},
  volume={},
  number={},
  pages={1-24},
  doi = {},
  }
```

## License
licensed for research use only. The dataset is CC BY NC 4.0 (allowing only non-commercial use), and models trained using the dataset should not be used outside of research purposes.

## Related resources
- [ASF-Dataset](https://search.asf.alaska.edu/)
- [SNAP Toolbox](http://step.esa.int/main/download/)
