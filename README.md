![demo](assets/DynamicEarth_logo.png)

<div align="center">

<h1>DynamicEarth: How Far are We from Open-Vocabulary Change Detection?</h1>

<div>
    <a href='https://likyoo.github.io/' target='_blank'>Kaiyu Li</a><sup>1</sup>&emsp;
    <a href='https://gr.xjtu.edu.cn/en/web/caoxiangyong' target='_blank'>Xiangyong Cao</a><sup>✉1</sup>&emsp;
    <a href='https://github.com/BLING-1994' target='_blank'>Yupeng Deng</a><sup>2</sup>&emsp;
    <a href='https://github.com/fitzpchao' target='_blank'>Chao Pang</a><sup>3</sup>&emsp;
    <a href='https://github.com/xcarl1' target='_blank'>Zepeng Xin</a><sup>1</sup>&emsp;
    <a href='' target='_blank'>Hui Qiao</a><sup>4</sup>&emsp;
    <a href='https://gong-tl.github.io/' target='_blank'>Tieliang Gong</a><sup>1</sup>&emsp;
    <a href='https://gr.xjtu.edu.cn/en/web/dymeng' target='_blank'>Deyu Meng</a><sup>1</sup>&emsp;
    <a href='https://gr.xjtu.edu.cn/en/web/zhiwang' target='_blank'>Zhi Wang</a><sup>1</sup>&emsp;
</div>
<div>
    <sup>1</sup>Xi'an Jiaotong University&emsp;
    <sup>2</sup>Chinese Academy of Sciences&emsp;
    <sup>3</sup>Wuhan University&emsp;
    <sup>4</sup>China Telecom&emsp;
</div>

<div>
    <h4 align="center">
        • <a href="https://likyoo.github.io/DynamicEarth/" target='_blank'>[Project]</a> • <a href="https://arxiv.org/abs/2501.12931" target='_blank'>[arXiv]</a> • <a href="" target='_blank'>[Colab]</a> •
    </h4>
</div>

<img src="https://github.com/user-attachments/assets/0530a114-e320-46a4-be0d-201c6dc22743" width="90%"/>
Different change detection tasks: (a) Binary change detection aims at discovering all (interested) changes and generating a binary mask; (b) Semantic change detection further identifies the category of changes. However, both can only be trained and evaluated on data with predefined categories; (c) Our proposed OVCD can detect changes in any category according to the user's requirements.

</div>

----

**【地表最强AI侦探上线！DynamicEarth：让遥感图像图像变化检测秒变"大家来找茬"Pro Max版🌍🔍】**

各位看官！还在为传统变化检测模型"死记硬背"有限类别而头秃吗？我们打造的开放词汇变化检测（OVCD）黑科技，让AI秒变"火眼金睛"——无需996式训练，直接调用现成基础模型，就能在卫星图上玩转"大家来找茬"！

👉 两大绝招横扫江湖：

1️⃣ **​M-C-I框架**："先圈地再破案"模式——SAM模型像撒网捕鱼般圈出可疑区域，DINO化身福尔摩斯比对特征，最后CLIP大佬开口定罪名："报告！这里从工地变泳池了！🏗→🏊"

2️⃣ **​I-M-C框架**："指哪打哪"模式——Grounding DINO先锁定目标："给我盯死这片别墅区！" SAM立刻画出精确轮廓，DINO翻出历史档案对比："老板，3号楼偷偷加盖了两层！"

💡 五大杀手锏：

✔️ 开放词汇任你撩：从"查违章建筑"到"找新开体育场"，输入文字指令就能精准定位

✔️ 零训练开箱即用：告别炼丹式调参，现有模型直接"拼积木"

✔️ 抗干扰能力MAX：光照变化？季节更替？我们的AI侦探绝不"疑神疑鬼"

✔️ 跨数据集乱杀：在LEVIR-CD等五大擂台赛吊打传统方法，F1分数飙升30%+

✔️ 代码全家桶奉上：DynamicEarth开源库已就位，就差你来Star⭐️


----

**"DynamicEarth: Where Satellite Sleuthing Meets Open-World Wizardry!"** 🌍🕵️♂️

Calling all geo-detectives! Tired of change detection models stuck in "I-Spy-20-Objects" mode? Meet our ​**Open-Vocabulary Change Detection (OVCD)** – the Sherlock Holmes of satellite imagery that cracks any visual case you throw at it, ​zero training required!

🚀 **​Two Frameworks to Rule Them All:**

1️⃣ **​M-C-I Protocol**: "Mask first, ask later!"

- **​SAM** sprays "detective spray" to highlight suspicious zones 🕸️
- **DINO** plays spot-the-difference with NASA-level precision 🔍
- **CLIP** drops the mic: "This construction site just morphed into a waterpark!" 🏗️💦

2️⃣ **​I-M-C** Maneuver: "Name it, claim it!"

- Point at a target: "Track every swimming pool in Dubai!" 🏊♂️

- **​Grounding DINO** snaps to attention 👮♂️

- **​SAM** outlines targets like a crime scene investigator 🚧

- **​DINO** cross-examines timelines: "Pool #5 shrank 2 meters – violation alert!" 🚨

💥 ​Why This Rocks:

✔️ **​Vocabulary? We Don’t Know Her:** Detect "illegal rooftop extensions" or "mysterious crop circles" with equal flair 🌾👽

✔️ **​No-Training Wheels:** Skip endless training marathons – our model’s already bench-pressing foundation models 💪

✔️ **​Pseudo-Change? GTFO:** Seasons change? Shadows shift? Our AI’s got trust issues (in a good way) ☀️❄️

✔️ **​Dataset Domination:** Crushed LEVIR-CD/WHU-CD benchmarks like Godzilla in Tokyo 🏙️💥

✔️ **​Open-Source Swagger:** DynamicEarth codebase – now 100% less "secret sauce"! 👩💻🔓


----

<div align="center">

<img src="https://github.com/user-attachments/assets/94b58131-4593-415b-9e44-0ee790f884ef" width="90%"/>

The two OVCD frameworks proposed in this paper. (a) M-C-I: discover all class-agnostic masks, determine if the mask region has changed, and identify the change class. (b) I-M-C: identify all targets of interest, convert to mask format, and compare if the target has changed.

</div>

## Abstract

Monitoring Earth's evolving land covers requires methods capable of detecting changes across a wide range of categories and contexts. Existing change detection methods are hindered by their dependency on predefined classes, reducing their effectiveness in open-world applications. To address this issue, we introduce open-vocabulary change detection (OVCD), a novel task that bridges vision and language to detect changes across any category. Considering the lack of high-quality data and annotation, we propose two training-free frameworks, M-C-I and I-M-C, which leverage and integrate off-the-shelf foundation models for the OVCD task. The insight behind the M-C-I framework is to discover all potential changes and then classify these changes, while the insight of I-M-C framework is to identify all targets of interest and then determine whether their states have changed. Based on these two frameworks, we instantiate to obtain several methods, e.g., SAM-DINOv2-SegEarth-OV, Grounding-DINO-SAM2-DINO, etc. Extensive evaluations on 5 benchmark datasets demonstrate the superior generalization and robustness of our OVCD methods over existing supervised and unsupervised methods. To support continued exploration, we release DynamicEarth, a dedicated codebase designed to advance research and application of OVCD.

## Dependencies and Installation

Our code depends on [PyTorch](https://pytorch.org/), [Detectron](https://github.com/facebookresearch/detectron2), [OpenMMLab](https://github.com/open-mmlab), [SAM](https://github.com/facebookresearch/segment-anything) ... ... 

Please refer to [Install Guide](install.md) for more detailed instruction.

## Demo

SAM_DINO_SegEarth-OV
```
python sam_dino_segearth-ov_demo.py --input_image_1 demo_images/A/test_1024.png --input_image_2 demo_images/B/test_1024.png
```

SAM_DINOv2_SegEarth-OV
```
python sam_dinov2_segearth-ov_demo.py --input_image_1 demo_images/A/test_1024.png --input_image_2 demo_images/B/test_1024.png
```

Grounding DINO 1.5-SAM2-DINO
```
# Get your API token from https://cloud.deepdataspace.com
python gd1.5_sam2_demo.py --gd_api_token [YOUR_TOKEN] --input_image_1 demo_images/A/test_256.png --input_image_2 demo_images/B/test_256.png 
```

APE-DINO
```
python ape_dino_demo.py --input_image_1 demo_images/A/test_256.png --input_image_2 demo_images/B/test_256.png 
```

APE-DINOv2
```
python ape_dinov2_demo.py --input_image_1 demo_images/A/test_256.png --input_image_2 demo_images/B/test_256.png 
```

MMGrounding DINO-SAM2-DINO
```
python mmgd_sam2_dino_demo.py --input_image_1 demo_images/A/test_256.png --input_image_2 demo_images/B/test_256.png 
```

## Evaluation

We provide comprehensive evaluation scripts for the [LEVIR-CD](https://justchenhao.github.io/LEVIR/), [WHU-CD](http://gpcv.whu.edu.cn/data/building_dataset.html), [S2Looking](https://github.com/S2Looking/Dataset), [BANDON](https://github.com/fitzpchao/BANDON), [SECOND](https://captain-whu.github.io/SCD/) datasets and you can find them in [eval](eval).

## Results

<div align="center">

<div>
<img src="https://github.com/user-attachments/assets/fa5ccb8e-cb59-447f-87b8-2caf30e8e5ee" width="70%"/>
</div>

<div>
<img src="https://github.com/user-attachments/assets/491156f3-ecd8-47b7-bc03-c5aa37c33e96" width="70%"/>
</div>

<div>
<img src="https://github.com/user-attachments/assets/2e11f37a-4a89-4e3c-997c-af3cb42ae290" width="70%"/>
</div>
</div>

## Visualization

<div>
<img src="https://github.com/user-attachments/assets/5c368d37-862d-4f6e-9702-c0bce3d48fba" width="100%"/>
</div>



## Citation

```
@article{li2025dynamicearth,
  title={DynamicEarth: How Far are We from Open-Vocabulary Change Detection?},
  author={Li, Kaiyu and Cao, Xiangyong and Deng, Yupeng and Pang, Chao and Xin, Zepeng and Meng, Deyu and Wang, Zhi},
  journal={arXiv preprint arXiv:2501.12931},
  year={2025}
}
```

## Acknowledgement

We sincerely appreciate the following:

- [AngChange](https://github.com/Z-Zheng/pytorch-change-models/tree/main/torchange/models/segment_any_change)
- [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
- [UCD-SCM](https://github.com/StephenApX/UCD-SCM)

