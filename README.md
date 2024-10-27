### FreeStyle: 使用扩散模型进行文本引导的风格迁移

#### 概述
生成扩散模型的快速发展极大地推动了风格迁移领域。然而，大多数基于扩散模型的现有方法通常涉及缓慢的迭代优化过程，例如模型微调和风格概念的文本反转。在本文中，我们介绍了 **FreeStyle**，这是一种基于预训练大型扩散模型的创新风格迁移方法，无需进一步优化。此外，我们的方法仅通过所需风格的文本描述实现风格迁移，无需风格图像。具体来说，我们提出了一种双流编码器和单流解码器架构，取代了扩散模型中的传统 U-Net。在双流编码器中，两个不同的分支分别以内容图像和风格文本提示作为输入，实现了内容和风格的解耦。在解码器中，我们进一步根据给定的内容图像和相应的风格文本提示对来自双流的功能进行调制，以实现精确的风格迁移。

更多详情请参见 [项目页面](https://freestylefreelunch.github.io/) 和 [论文](https://arxiv.org/pdf/2401.15636.pdf)。

### 快速开始

#### 前提条件

```sh
conda create -n stylefree python==3.8.18
conda activate stylefree
```

#### 安装

安装依赖项

```sh
sudo apt-get update && sudo apt-get install git-lfs ffmpeg cbm
git clone https://github.com/FreeStyleFreeLunch/FreeStyle && cd FreeStyle/diffusers
pip install -e .
pip install torchsde -i https://pypi.tuna.tsinghua.edu.cn/simple
cd ../diffusers_test
pip install transformers accelerate huggingface_hub==0.25.0
```

#### 下载模型和权重文件

下载 [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main) 并将其放入：**./diffusers_test/stable-diffusion-xl-base-1.0**

```sh
git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
du -sh stable-diffusion-xl-base-1.0
```

### 示例

你可以在路径：`./diffusers_test/ContentImages/imgs_and_hyperparameters/` 中找到一些示例及其特定的参数设置。你可以通过设置自己的任务来运行它们。此外，你可以使用以下代码快速运行一个示例。

#### 油画风格

```sh
cd ./diffusers_test
python stable_diffusion_xl_test.py --refimgpath ./ContentImages/imgs0 --model_name "./stable-diffusion-xl-base-1.0" --unet_name ./stable-diffusion-xl-base-1.0/unet/ --prompt_json ./style_prompt0.json --num_images_per_prompt 4 --output_dir ./output0 --sampler "DDIM" --step 30 --cfg 5 --height 1024 --width 1024 --seed 123456789 --n 160 --b 1.8 --s 1
```

#### 折纸艺术风格

```sh
cd ./diffusers_test
python stable_diffusion_xl_test.py --refimgpath ./ContentImages/imgs1 --model_name "./stable-diffusion-xl-base-1.0" --unet_name ./stable-diffusion-xl-base-1.0/unet/ --prompt_json ./style_prompt1.json --num_images_per_prompt 4 --output_dir ./output1 --sampler "DDIM" --step 30 --cfg 5 --height 1024 --width 1024 --seed 123456789 --n 160 --b 2.5 --s 1
```

#### 梵高星空风格

```sh
cd ./diffusers_test
python stable_diffusion_xl_test.py --refimgpath ./ContentImages/imgs2 --model_name "./stable-diffusion-xl-base-1.0" --unet_name ./stable-diffusion-xl-base-1.0/unet/ --prompt_json ./style_prompt2.json --num_images_per_prompt 4 --output_dir ./output2 --sampler "DDIM" --step 30 --cfg 5 --height 1024 --width 1024 --seed 123456789 --n 160 --b 2.5 --s 1
```

#### 吉卜力工作室风格

```sh
cd ./diffusers_test
python stable_diffusion_xl_test.py --refimgpath ./ContentImages/imgs3 --model_name "./stable-diffusion-xl-base-1.0" --unet_name ./stable-diffusion-xl-base-1.0/unet/ --prompt_json ./style_prompt3.json --num_images_per_prompt 4 --output_dir ./output3 --sampler "DDIM" --step 30 --cfg 5 --height 1024 --width 1024 --seed 123456789 --n 160 --b 2.8 --s 1
```

#### 赛博朋克风格

```sh
cd ./diffusers_test
python stable_diffusion_xl_test.py --refimgpath ./ContentImages/imgs4 --model_name "./stable-diffusion-xl-base-1.0" --unet_name ./stable-diffusion-xl-base-1.0/unet/ --prompt_json ./style_prompt4.json --num_images_per_prompt 4 --output_dir ./output4 --sampler "DDIM" --step 30 --cfg 5 --height 1024 --width 1024 --seed 123456789 --n 160 --b 2.8 --s 1
```

#### 儿童蜡笔画风格

```sh
cd ./diffusers_test
python stable_diffusion_xl_test.py --refimgpath ./ContentImages/imgs5 --model_name "./stable-diffusion-xl-base-1.0" --unet_name ./stable-diffusion-xl-base-1.0/unet/ --prompt_json ./style_prompt5.json --num_images_per_prompt 4 --output_dir ./output5 --sampler "DDIM" --step 30 --cfg 5 --height 1024 --width 1024 --seed 123456789 --n 160 --b 1.8 --s 1
```

### 开始推理

根据以下方法进行风格迁移推理。

#### 推理命令

```sh
cd ./diffusers_test
python stable_diffusion_xl_test.py --refimgpath ./ContentImages/imgs0 --model_name "./stable-diffusion-xl-base-1.0" --unet_name ./stable-diffusion-xl-base-1.0/unet/ --prompt_json ./style_prompt0.json --num_images_per_prompt 4 --output_dir ./output1 --sampler "DDIM" --step 30 --cfg 5 --height 1024 --width 1024 --seed 123456789 --n 640 --b 1.5 --s 2
```

#### 参数解释

1. **--refimgpath**: 内容图像路径
   指定用于风格化的内容图像的路径。

2. **--model_name**: SDXL 模型保存路径
   指定保存 SDXL 模型的路径。

3. **--unet_name**: SDXL 中 unet 文件夹路径
   指定 SDXL 模型中 unet 文件夹的路径。

4. **--prompt_json**: 风格提示的 JSON 文件
   指定包含风格提示的 JSON 文件路径。

5. **--num_images_per_prompt**: 每个图像和风格生成多少张图像
   指定每个内容图像和风格组合生成多少张图像。

6. **--output_dir**: 保存风格化图像的路径
   指定保存生成的风格化图像的输出目录。

7. **--n**: 超参数 n∈[160,320,640,1280]
   指定超参数 n，其取值范围为 160、320、640 或 1280。

8. **--b**: 超参数 b∈(1,3)
   指定超参数 b，其取值范围为 1 到 3 之间的浮点数。

9. **--s**: 超参数 s∈(1,2)
   指定超参数 s，其取值范围为 1 到 2 之间的浮点数。

### 参数推荐

- **内容图像质量**: 高质量的内容图像可以获得更好的风格化效果。

- **推荐参数设置**:
  - `n=160`
  - `b=2.5`
  - `s=1`

- **风格信息表达模糊**: 当风格信息的表达不明确时，建议减小 `b`（增大 `s`）。

- **内容信息表达不清晰**: 当内容信息的表达不清晰时，建议增大 `b`（减小 `s`）。

- **风格化图像中的噪声**: 当风格化图像中存在噪声时，适当调整参数 `n`。

### 提示

- **图像比例**: 可以通过 `diffusers_test/centercrop.py` 获取 1:1 比例的图像。

- **超参数调整**: 对于大多数图像，调整超参数通常可以获得满意的结果。如果生成的结果不理想，建议尝试不同的超参数组合。

### 示例输出

#### 源图像

```python
from IPython import display
display.Image("ContentImages/imgs0/0000.png")
display.Image("ContentImages/imgs0/0001.png")
```

#### 风格提示

```python
import json
with open("style_prompt0.json", "r") as f:
    print(json.load(f))
# ['Oil painting style']
```

#### 目标图像

```python
display.Image("output0_fp16/0001/00000_00.png")
display.Image("output0_fp16/0001/00000_01.png")
```

### 测试结果

你可以在路径：`./diffusers_test/ContentImages/imgs_and_hyperparameters/` 中找到一些示例及其特定的参数设置。

#### 是否真实

```sh
cd ContentImages && mkdir genshin_impact_imgs0
```

### 不能随便设置

```python
with open("lego_tyo_prompt0.json", "w") as f:
    json.dump(["LEGO Toy"], f)

with open("chineseink.json", "w") as f:
    json.dump(["chineseink"], f)
```

#### 示例命令

```sh
python stable_diffusion_xl_test.py --refimgpath ./ContentImages/genshin_impact_imgs0 --model_name "./stable-diffusion-xl-base-1.0" --unet_name ./stable-diffusion-xl-base-1.0/unet/ --prompt_json ./lego_tyo_prompt0.json --num_images_per_prompt 4 --output_dir ./genshin_impact_output0 --sampler "DDIM" --step 30 --cfg 5 --height 1024 --width 1024 --seed 123456789 --n 160 --b 1.8 --s 1

python stable_diffusion_xl_test.py --refimgpath ./ContentImages/genshin_impact_imgs0 --model_name "./stable-diffusion-xl-base-1.0" \
 --unet_name ./stable-diffusion-xl-base-1.0/unet/ --prompt_json ./chineseink.json --num_images_per_prompt 1 \
 --output_dir ./genshin_impact_output1 --sampler "DDIM" --step 50 --cfg 5 --height 1024 --width 1024 --seed 123456789 --n 320 --b 2.8 --s 0.2

python stable_diffusion_xl_test.py --refimgpath ./ContentImages/imgs0 --model_name "./stable-diffusion-xl-base-1.0" \
  --unet_name ./stable-diffusion-xl-base-1.0/unet/ --prompt_json ./chineseink.json --num_images_per_prompt 1 \
  --output_dir ./img0_output_0 --sampler "DDIM" --step 50 --cfg 5 --height 1024 --width 1024 --seed 123456789 --n 320 --b 2.8 --s 0.2
```

### 引用

```sh
@misc{he2024freestyle,
  title={FreeStyle: Free Lunch for Text-guided Style Transfer using Diffusion Models},
  author={Feihong He and Gang Li and Mengyuan Zhang and Leilei Yan and Lingyu Si and Fanzhang Li},
  year={2024},
  eprint={2401.15636},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
