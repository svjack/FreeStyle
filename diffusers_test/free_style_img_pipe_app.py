import os
import shutil
import json
import gradio as gr
from glob import glob
import torch
from diffusers import (
    DPMSolverSDEScheduler,
    DDIMScheduler,
    UNet2DConditionModel
)
from pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from pipeline_stable_diffusion_img2img import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import math

HAVE_REFINER = False

# 创建临时目录
def create_temp_dirs():
    refimgpath = "./ContentImages/temp_imgs0"
    output_dir = "./temp_output0"
    if not os.path.exists(refimgpath):
        os.makedirs(refimgpath)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return refimgpath, output_dir

# 清空临时目录
def clear_temp_dirs(refimgpath, output_dir):
    if os.path.exists(refimgpath):
        shutil.rmtree(refimgpath)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

# 保存风格提示到 JSON 文件
def save_prompt_to_json(prompt, json_path):
    with open(json_path, "w") as f:
        json.dump([prompt], f)

# 初始化管道
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
noise_scheduler = DDIMScheduler.from_pretrained("stable-diffusion-xl-base-1.0", subfolder="scheduler")
unet = UNet2DConditionModel.from_pretrained("stable-diffusion-xl-base-1.0", subfolder='unet')
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stable-diffusion-xl-base-1.0", scheduler=noise_scheduler, unet=unet.to(dtype=torch.float16), torch_dtype=torch.float16
).to(device)
if HAVE_REFINER:
    pipeline_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stable-diffusion-xl-refiner-1.0"
    ).to(device)
else:
    pipeline_refiner = None

# 运行风格迁移
def run_style_transfer(image, style_image, prompt, num_images, sampler, step, cfg, height, width, seed, n, b, s, refiner, grid):
    refimgpath, output_dir = create_temp_dirs()

    if os.path.exists(os.path.join("{}_fp16".format(output_dir))):
        shutil.rmtree(os.path.join("{}_fp16".format(output_dir)))

    image_path = os.path.join(refimgpath, "content_image.png")
    image.save(image_path)
    image_path = os.path.join(refimgpath, "style_image.png")
    style_image.save(image_path)

    json_path = "style_prompt.json"
    save_prompt_to_json(prompt, json_path)

    output_dir = output_dir + '_fp16'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    files = os.listdir(refimgpath)
    files = [os.path.join(refimgpath, file) for file in files]
    img_files = list(filter(lambda x: "content_image" in x, files))
    sty_files = list(filter(lambda x: "style_image" in x, files))

    all_images = []

    class ARGS(object):
        def __init__(self, b, s, n):
            self.b = b
            self.s = s
            self.n = n
    args = ARGS(b, s, n)
    unet.set_hyper_parameter(args)

    for file in img_files:
        if refiner and pipeline_refiner is not None:
            if sty_files:
                print("use ", sty_files, "in refiner")
                outputs = pipeline(
                    prompt,
                    negative_prompt="worst quality, low quality, low res, blurry, cropped image, jpeg artifacts, error, ugly, out of frame, deformed, poorly drawn, mutilated, mangled, bad proportions, long neck, missing limb, floating limbs, disconnected limbs, long body, missing arms, malformed limbs, missing legs, extra arms, extra legs, poorly drawn face, cloned face, deformed iris, deformed pupils, deformed hands, twisted fingers, malformed hands, poorly drawn hands, mutated hands, mutilated hands, extra fingers, fused fingers, too many fingers, duplicate, multiple heads, extra limb, duplicate artifacts",
                    num_images_per_prompt=num_images,
                    width=width,
                    height=height,
                    num_inference_steps=step,
                    guidance_scale=cfg,
                    generator=torch.Generator(device=device).manual_seed(seed),
                    output_type="latent",
                    refimgpath=file,
                    styleimgpath=sty_files[0]
                ).images
            else:
                outputs = pipeline(
                    prompt,
                    negative_prompt="worst quality, low quality, low res, blurry, cropped image, jpeg artifacts, error, ugly, out of frame, deformed, poorly drawn, mutilated, mangled, bad proportions, long neck, missing limb, floating limbs, disconnected limbs, long body, missing arms, malformed limbs, missing legs, extra arms, extra legs, poorly drawn face, cloned face, deformed iris, deformed pupils, deformed hands, twisted fingers, malformed hands, poorly drawn hands, mutated hands, mutilated hands, extra fingers, fused fingers, too many fingers, duplicate, multiple heads, extra limb, duplicate artifacts",
                    num_images_per_prompt=num_images,
                    width=width,
                    height=height,
                    num_inference_steps=step,
                    guidance_scale=cfg,
                    generator=torch.Generator(device=device).manual_seed(seed),
                    output_type="latent",
                    refimgpath=file,
                    #styleimgpath=sty_files[0]
                ).images

            outputs = pipeline_refiner(
                prompt,
                image=outputs,
                negative_prompt="worst quality, low quality, low res, blurry, cropped image, jpeg artifacts, error, ugly, out of frame, deformed, poorly drawn, mutilated, mangled, bad proportions, long neck, missing limb, floating limbs, disconnected limbs, long body, missing arms, malformed limbs, missing legs, extra arms, extra legs, poorly drawn face, cloned face, deformed iris, deformed pupils, deformed hands, twisted fingers, malformed hands, poorly drawn hands, mutated hands, mutilated hands, extra fingers, fused fingers, too many fingers, duplicate, multiple heads, extra limb, duplicate artifacts",
                num_images_per_prompt=num_images,
                num_inference_steps=step,
                guidance_scale=cfg,
                generator=torch.Generator(device=device).manual_seed(seed),
                refimgpath=file,
            )
        else:
            if sty_files:
                print("use ", sty_files, "in pipe")
                outputs = pipeline(
                    prompt,
                    negative_prompt="worst quality, low quality, low res, blurry, cropped image, jpeg artifacts, error, ugly, out of frame, deformed, poorly drawn, mutilated, mangled, bad proportions, long neck, missing limb, floating limbs, disconnected limbs, long body, missing arms, malformed limbs, missing legs, extra arms, extra legs, poorly drawn face, cloned face, deformed iris, deformed pupils, deformed hands, twisted fingers, malformed hands, poorly drawn hands, mutated hands, mutilated hands, extra fingers, fused fingers, too many fingers, duplicate, multiple heads, extra limb, duplicate artifacts",
                    num_images_per_prompt=num_images,
                    width=width,
                    height=height,
                    num_inference_steps=step,
                    guidance_scale=cfg,
                    generator=torch.Generator(device=device).manual_seed(seed),
                    #output_type="latent",
                    refimgpath=file,
                    styleimgpath=sty_files[0]
                )
            else:
                outputs = pipeline(
                    prompt,
                    negative_prompt="worst quality, low quality, low res, blurry, cropped image, jpeg artifacts, error, ugly, out of frame, deformed, poorly drawn, mutilated, mangled, bad proportions, long neck, missing limb, floating limbs, disconnected limbs, long body, missing arms, malformed limbs, missing legs, extra arms, extra legs, poorly drawn face, cloned face, deformed iris, deformed pupils, deformed hands, twisted fingers, malformed hands, poorly drawn hands, mutated hands, mutilated hands, extra fingers, fused fingers, too many fingers, duplicate, multiple heads, extra limb, duplicate artifacts",
                    num_images_per_prompt=num_images,
                    width=width,
                    height=height,
                    num_inference_steps=step,
                    guidance_scale=cfg,
                    generator=torch.Generator(device=device).manual_seed(seed),
                    #output_type="latent",
                    refimgpath=file,
                    #styleimgpath=sty_files[0]
                )

        images = outputs.images
        all_images.extend(images)

    to_save_images = []
    if grid:
        to_save_width = int(math.sqrt(num_images))
        to_save_height = num_images // to_save_width
        if to_save_height * to_save_width != num_images:
            to_save_width += 1
        for _idx in range(0, len(all_images), num_images):
            new_img = Image.new("RGB", (to_save_width * width, to_save_height * height))
            for subIdx in range(num_images):
                x_offset = (subIdx % to_save_width) * width
                y_offset = (subIdx // to_save_height) * height
                new_img.paste(all_images[_idx + subIdx], (x_offset, y_offset))
            to_save_images.append(new_img)
    else:
        to_save_images = all_images

    '''
    for id, image in enumerate(to_save_images):
        image.save(os.path.join(output_dir, f'{id:05d}.png'))
    to_save_images = list(map(lambda x: x.cpu() if hasattr(x, "cpu") else x, to_save_images))
    to_save_images = list(map(lambda x: x.numpy() if hasattr(x, "numpy") else x, to_save_images))
    print(to_save_images)
    import pickle as pkl
    with open("arr.pkl", "wb") as f:
        pkl.dump(to_save_images, f)
    '''

    clear_temp_dirs(refimgpath, output_dir)
    return to_save_images

# Gradio 接口
def gradio_interface(image, style_image, prompt, num_images, sampler, step, cfg, height, width, seed, n, b, s, refiner, grid):
    generated_images = run_style_transfer(image, style_image, prompt, num_images, sampler, step, cfg, height, width, seed, n, b, s, refiner, grid)
    return generated_images

# 示例数据
examples = [[
'ContentImages/imgs0/0000.png',
'ContentImages/imgs1/0000.png',
  'Cyberpunk',
  1,
  'DDIM',
  50,
  5,
  1024,
  1024,
  123456789,
  160,
  2.8,
  1.1,
  False,
  False]
]


# Gradio 应用
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Image(type="pil", label = "Content Image"),
        gr.Image(type="pil", label = "Style Image"),
        gr.Textbox(label="Style Prompt"),
        gr.Slider(label="Number of Images per Prompt", minimum=1, maximum=10, step=1, value=1),
        gr.Dropdown(label="Sampler", choices=["DPM++ SDE Karras", "DDIM"], value="DDIM"),
        gr.Slider(label="Step", minimum=1, maximum=100, step=1, value=50),
        gr.Slider(label="CFG", minimum=1, maximum=10, step=1, value=5),
        gr.Slider(label="Height", minimum=256, maximum=2048, step=256, value=1024),
        gr.Slider(label="Width", minimum=256, maximum=2048, step=256, value=1024),
        gr.Number(label="Seed", value=123456789),
        gr.Number(label="n", value=160),
        gr.Number(label="b", value=2.5),
        gr.Number(label="s", value=1.0),
        gr.Checkbox(label="Refiner", value=False),
        gr.Checkbox(label="Grid", value=False)
    ],
    outputs=gr.Gallery(label="Generated Images"),
    title="FreeStyle: Text-guided Style Transfer using Diffusion Models",
    description="Upload an image and enter a style prompt to generate stylized images.",
    examples=examples
)

# 运行 Gradio 应用
iface.launch(share=True)
