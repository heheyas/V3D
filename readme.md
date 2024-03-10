# V3D: Video Diffusion Models are Effective 3D Generators
Zilong Chen, Yikai Wang, Feng Wang, Zhengyi Wang, Huaping Liu
Tsinghua University

This repository contains the official implementation of [V3D: Video Diffusion Models are Effective 3D Generators](404). 


### [Paper](TBD) | [Project Page](TBD) | [HF Demo](TBD)

### Video results
Single Image to 3D

Sparse view scene generation (On CO3D and MVImgNet)


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
