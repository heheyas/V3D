# V3D: Video Diffusion Models are Effective 3D Generators
Zilong Chen, Yikai Wang, Feng Wang, Zhengyi Wang, Huaping Liu

Tsinghua University, ShengShu

This repository contains the official implementation of [V3D: Video Diffusion Models are Effective 3D Generators](404). 


### [Paper](TBD) | [Project Page](TBD) | [HF Demo](TBD)

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
wget xxxx ckpts/
```
3. Run the V3D Video diffusion to generate dense multi-views
```
pass
```
4. Reconstruct 3D assets from generated multi-views
Using 3D Gaussian Splatting
```
python recon/train_from_vid.py
```
Or using (NeuS) instant-nsr-pl:
```
python mesh-recon/launch.py
```

## Acknowledgement
This code base is built upon the following awesome open-source projects:
- [Stable Video Diffusion](https://github.com/ashawkey/stable-dreamfusion)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [kiuikit](https://github.com/openai/shap-e)

Thank the authors for their remarkable job !
