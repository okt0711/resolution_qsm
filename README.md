# Unsupervised resolution-agnostic quantitative susceptibility mapping using adaptive instance normalization

This repository is the official pytorch implementation of "Unsupervised resolution-agnostic quantitative susceptibility mapping using adaptive instance normalization".

> "Unsupervised resolution-agnostic quantitative susceptibility mapping using adaptive instance normalization",  
> Gyutaek Oh, Hyokyoung Bae, Hyun-Seo Ahn, Sung-Hong Park, Won-Jin Moon, and Jong Chul Ye,  
> Medical Image Analysis, 2022 [[Paper]](https://www.sciencedirect.com/science/article/pii/S1361841522001244?casa_token=Ow6f9DXE0A0AAAAA:7LcEjR46Ouxj_EM1dykee4VH7nUvW_I_KBUF8lp3zpqSQPIWt2BBxYNpjJoHf3og1MHIWFkIQg)

## Requirements
The code is implented in Python 3.7 with below packages.
```
torch               1.8.1
numpy               1.21.6
scipy               1.7.3
```

## Training and Inference
To evaluate, run the below commands.
```
sh run.sh
```
To train the model, add the ```--training``` option in the script files.

## Citation
If you find our work interesting, please consider citing
```
@article{oh2022unsupervised,
  title={Unsupervised resolution-agnostic quantitative susceptibility mapping using adaptive instance normalization},
  author={Oh, Gyutaek and Bae, Hyokyoung and Ahn, Hyun-Seo and Park, Sung-Hong and Moon, Won-Jin and Ye, Jong Chul},
  journal={Medical Image Analysis},
  volume={79},
  pages={102477},
  year={2022},
  publisher={Elsevier}
}
```
