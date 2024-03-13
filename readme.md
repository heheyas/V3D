# V3D: Video Diffusion Models are Effective 3D Generators
Zilong Chen<sup>1,2</sup>, Yikai Wang<sup>1</sup>, Feng Wang<sup>1</sup>, Zhengyi Wang<sup>1,2</sup>, Huaping Liu<sup>1</sup>

<sup>1</sup>Tsinghua University, <sup>2</sup>ShengShu

This repository contains the official implementation of [V3D: Video Diffusion Models are Effective 3D Generators](https://arxiv.org/abs/2403.06738). 

### What's New
[2024.3.14] Our demo is currently available at (here)[https://huggingface.co/spaces/heheyas/V3D]. I will add more checkpoints and more examples recently.

### [Work in Progress]

We are currently working on making this completely publicly available (including refactoring code, uploading weights, etc.), so please be patient.

### [arXiv](https://arxiv.org/abs/2403.06738) | [Paper](assets/pdf/V3D.pdf) | [Project Page](https://heheyas.github.io/V3D) | [HF Demo](https://huggingface.co/spaces/heheyas/V3D)

### Video results
Single Image to 3D

Generated Multi-views

https://github.com/heheyas/V3D/assets/44675551/bb724ed1-b9a6-4aa7-9a49-f1a8c8756c2f


https://github.com/heheyas/V3D/assets/44675551/4bfaea91-6c5b-40da-8682-30286a916979

Reconstructed 3D Gaussian Splats


https://github.com/heheyas/V3D/assets/44675551/894444eb-a454-4bc9-921b-cd0d5764a14d



https://github.com/heheyas/V3D/assets/44675551/eda05891-e2c7-4f44-af12-9ccd0bce61d1



https://github.com/heheyas/V3D/assets/44675551/27d61245-b416-4289-ba98-97219ad199a3



https://github.com/heheyas/V3D/assets/44675551/e94d71ff-b8bc-410c-ad2c-3cfb1fbef7fa



https://github.com/heheyas/V3D/assets/44675551/a0d1e971-0f8f-4f05-a73e-45271e37a31f



https://github.com/heheyas/V3D/assets/44675551/0dac3189-fc59-4e9b-8151-10ebe2711d71


Sparse view scene generation (On CO3D `hydrant` category)


https://github.com/heheyas/V3D/assets/44675551/33c87468-b6c0-4fa2-a9bf-6f396b3fa089


https://github.com/heheyas/V3D/assets/44675551/3c03d015-2e56-44de-8210-e33e7ec810bb



https://github.com/heheyas/V3D/assets/44675551/1e73958b-04b2-4faa-bbc3-675399f21956



https://github.com/heheyas/V3D/assets/44675551/f70cc259-7d50-4bf9-9c1b-0d4143ae8958



https://github.com/heheyas/V3D/assets/44675551/f6407b02-5ee7-4f8f-8559-4a893e6fd912





### Instructions:
1. Install the requirements:
```
pip install -r requirements.txt
```
2. Download our weights for V3D
```
wget https://huggingface.co/heheyas/V3D/resolve/main/V3D.ckpt -O ckpts/V3D_512.ckpt
wget https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt.safetensors -O ckpts/svd_xt.safetensors
```
3. Run the V3D Video diffusion to generate dense multi-views
```
PYTHONPATH="." python scripts/pub/V3D_512.py --input_path <image file or dir> --save --border_ratio 0.3 --min_guidance_scale 4.5 --max_guidance_scale 4.5 --output-folder <output-dest>
```
4. Reconstruct 3D assets from generated multi-views
Using 3D Gaussian Splatting
```
PYTHONPATH="." python recon/train_from_vid.py  -w --sh_degree 0 --iterations 4000 --lambda_dssim 1.0 --lambda_lpips 2.0 --save_iterations 4000 --num_pts 100_000 --video <your generated video>
```
Or using (NeuS) instant-nsr-pl:
```
cd mesh_recon
PYTHONPATH="." python launch.py --config configs/videonvs.yaml --gpu <gpu> --train system.loss.lambda_normal=0.1 dataset.scene=<scene_name> dataset.root_dir=<output_dir> dataset.img_wh='[512, 512]'
```
Refine texture
```
python refine.py --mesh <your obj mesh file> --scene <your video> --num-opt 16 --lpips 1.0 --iters 500
```

## Acknowledgement
This code base is built upon the following awesome open-source projects:
- [Stable Video Diffusion](https://github.com/Stability-AI/generative-models)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [kiuikit](https://github.com/ashawkey/kiuikit)
- [Instant-nsr-pl](https://github.com/bennyguo/instant-nsr-pl)

Thank the authors for their remarkable job !
