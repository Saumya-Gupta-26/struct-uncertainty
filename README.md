# Topology-Aware Uncertainty for Image Segmentation
This repository contains the implementation for our work "[Topology-Aware Uncertainty for Image Segmentation](https://arxiv.org/abs/2306.05671)", accepted to NeurIPS 2023. 

The `dmt-uncertainty.ipynb` file contains simple code on how to use discrete Morse theory (DMT) to generate the Morse Skeleton (and the corresponding critical points and manifolds). It has some visualizations to understand the main code better.

## USAGE
The current code is for 2D images. You will notice the word 'DRIVE' everywhere --- this is the dataset I used. You need to replace it with your dataset.

### Dipha
Run the following commands. Need to run this only once.

```
cd dipha-graph-recon/
rm -rf build/
mkdir build
cd build
cmake ..
make
```

### Training
- Edit `datalists/DRIVE/train.json` with your hyperparameter values.
- Command to run: `CUDA_VISIBLE_DEVICES=7 python3 train.py --params ./datalists/DRIVE/train.json`

### Inference
- Edit `datalists/DRIVE/infer.json` with your hyperparameter values.
- Command to run: `CUDA_VISIBLE_DEVICES=7 python3 infer.py --params ./datalists/DRIVE/infer.json`

## Acknowledgement
The code for computing DMT has been borrowed from [here](https://github.com/wangdingkang/DiscreteMorse) . I would like to thank them because it has formed the basis of this work. I modified their code to output the generated manifolds.

## CITATION
If you found this work useful, please consider citing it as
```
@article{gupta2024topology,
  title={Topology-aware uncertainty for image segmentation},
  author={Gupta, Saumya and Zhang, Yikai and Hu, Xiaoling and Prasanna, Prateek and Chen, Chao},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```
