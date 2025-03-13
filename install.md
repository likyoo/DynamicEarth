### DynamicEarth Inrstallation Guide

1. Check the nvcc version

```
nvcc --version
```

If the above command outputs the following message, it means that the nvcc setting is OK, otherwise you need to set CUDA_HOME.

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Jun_13_19:16:58_PDT_2023
Cuda compilation tools, release 12.2, V12.2.91
Build cuda_12.2.r12.2/compiler.32965470_0
```

2. Check the gcc version (requires 5.4+)

```
gcc --version
```

3. Install APE related dependencies

```
# Please refer to the official website (https://github.com/shenyunhang/APE).
# For instance, CUDA Version == 12.2
pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
git+https://github.com/facebookresearch/detectron2@main
git+https://github.com/IDEA-Research/detrex@main
git+https://github.com/openai/CLIP.git@main
pip install transformers==4.32.1 einops lvis
wget https://hf-mirror.com/shenyunhang/APE/resolve/main/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4_1080k_mdl_20230829_162438/model_final.pth
wget https://hf-mirror.com/QuanSun/EVA-CLIP/resolve/main/EVA02_CLIP_E_psz14_plus_s9B.pt

cd third_party
git clone https://github.com/shenyunhang/APE
cd APE
pip install -e .
```

4. Install openmmlab related dependencies

```
# install mmcv
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
pip install -r requirements/optional.txt
pip install -e . -v
```

```
# install mmengine
git clone https://github.com/open-mmlab/mmengine.git
cd mmengine
pip install -e . -v
```

```
# install mmsegmentation
# install mmdetection
```

5. Install DDS cloudapi (optional)

```
pip install dds-cloudapi-sdk --upgrade
```

6. Install SAM and SAM2

```
cd third_party/segment_anything
pip install -e .

cd third_party/sam2
pip install -e .
```

7. Install SegEarth-OV

```
cd third_party/SimFeatUp
pip install -e .
```

