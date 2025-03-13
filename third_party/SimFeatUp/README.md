<div align="center">

<h1>SegEarth-OV: Towards Training-Free Open-Vocabulary Segmentation for Remote Sensing Images</h1>

<div>
    <strong>Make OVSS possible in remote sensing contexts</strong>
</div>

<div>
    <a href='https://likyoo.github.io/' target='_blank'>Kaiyu Li</a><sup>1</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=WTleRV8AAAAJ' target='_blank'>Ruixun Liu</a><sup>1</sup>&emsp;
    <a href='https://gr.xjtu.edu.cn/en/web/caoxiangyong' target='_blank'>Xiangyong Cao</a><sup>✉1</sup>&emsp;
    <a href='https://web.xidian.edu.cn/xrbai' target='_blank'>Xueru Bai</a><sup>2</sup>&emsp;
    <a href='https://faculty.xidian.edu.cn/ZF3' target='_blank'>Feng Zhou</a><sup>2</sup>&emsp;
    <a href='https://gr.xjtu.edu.cn/en/web/dymeng' target='_blank'>Deyu Meng</a><sup>1</sup>&emsp;
    <a href='https://gr.xjtu.edu.cn/en/web/zhiwang' target='_blank'>Zhi Wang</a><sup>1</sup>&emsp;
</div>
<div>
    <sup>1</sup>Xi'an Jiaotong University&emsp;
    <sup>2</sup>Xidian University&emsp;
</div>

<div>
    <h4 align="center">
        • <a href="https://likyoo.github.io/SegEarth-OV/" target='_blank'>[Project]</a> • <a href="https://arxiv.org/abs/2410.01768" target='_blank'>[arXiv]</a> • <a href="https://colab.research.google.com/drive/1a-NNz_2maesvszk4Xff5PKY02_moPqt6#scrollTo=Pz9QGEcFBGtK" target='_blank'>[Colab]</a> •
    </h4>
</div>

<img src="https://github.com/user-attachments/assets/5cf146c3-b719-489d-a9bf-0700fb72c196" width="100%"/>
</div>

Illustration of the proposed method. (a) is the training process of SimFeatUp. CLIP is frozen and only SimFeatUp is useful in reasoning. (b) is the reasoning process of SegEarth-OV. The LR feature maps from CLIP are upsampled by SimFeatUp and then the \texttt{[CLS]} token is subtracted to alleviate global bias. For better presentation, the color renderings follow [FeatUp](https://github.com/mhamilton723/FeatUp).

------

This is a sub-project of [SegEarth-OV](https://github.com/likyoo/SegEarth-OV) to **build** a CUDA version of JBU operation (provided by [FeatUp](https://github.com/mhamilton723/FeatUp)) and **train** SimFeatUp.

## Install

### Pip
For those just looking to quickly use the FeatUp APIs install via:
```shell script
pip install git+https://github.com/likyoo/SimFeatUp.git
```

### Local Development
To install FeatUp for local development and to get access to the sample images install using the following:
```shell script
git clone https://github.com/likyoo/SimFeatUp.git
cd FeatUp
pip install -e .
```

### Using Docker (recommended)
Since building CUDA arithmetic often encounters errors, we provide a Docker image.
```shell script
cd docker
# build docker image
docker image build -t segearth:v1 -f Dockerfile .
# run a docker, e.g.
docker run --name segearth --shm-size=4g --gpus all -it segearth:v1 bash
# see the docker documentation for more detailed usage
```

------

If you just want to build the CUDA version of JBU operation for SegEarth-OV inference, that's the end of it here.

If you want to train a your own SimFeatUp, you'll need to download the Million-AID dataset.

## Datasets

[Million-AID Download (One Drive)](https://whueducn-my.sharepoint.com/:f:/g/personal/longyang_whu_edu_cn/Et-SJsQYQRxMh63Z59iFyH0BramZuLnyo4XKoi5yrbfb9A) | [Million-AID Download (Baidu Drive, extraction code: 107t)](https://pan.baidu.com/s/1URt_dyAybExu9fsOg4lZgQ)

More details about Million-AID can be found [here](https://captain-whu.github.io/DiRS).

## Train

Change ``pytorch_data_dir`` in the [config](https://github.com/likyoo/SimFeatUp/blob/main/featup/configs/upsampler_aid.yaml) file to your data path. We use 2 GPUs with a batch size of 4 on each GPU by default.

```
python featup/train_upsampler.py
```

## Citation

```
@article{li2024segearth,
  title={SegEarth-OV: Towards Training-Free Open-Vocabulary Segmentation for Remote Sensing Images},
  author={Li, Kaiyu and Liu, ruixun and Cao, Xiangyong and Bai, Xueru and Zhou, Feng and Meng, Deyu and Wang, Zhi},
  journal={arXiv preprint arXiv:2410.01768},
  year={2024}
}
```

## Acknowledgement
This implementation is based on [FeatUp](https://github.com/mhamilton723/FeatUp). Thanks for the awesome work.
