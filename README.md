# SinGAN
This is an unofficial implementation of SinGAN from someone who's been sitting right next to SinGAN's creator for almost five years.


<p align="center">
  <img width="992" height="372" src="/figures/sampled.png">
</p>


Please refer the project's [page](https://tamarott.github.io/SinGAN.htm) for more details.



## Citation
If you use this code for your research, please cite the paper:

```
@inproceedings{shaham2019singan,
  title={Singan: Learning a generative model from a single natural image},
  author={Shaham, Tamar Rott and Dekel, Tali and Michaeli, Tomer},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4570--4580},
  year={2019}
}
```


## Code

### Clone repository

Clone this repository into any place you want.

```
git clone https://github.com/kligvasser/SinGAN
cd ./SinGAN/generation/
```

### Install dependencies

```
python -m pip install -r requirements.txt
```

This code tested in PyTorch 1.8.1.

### Training
To train SinGAN model on your own image:

```
python3 main.py --root <path-to-image>
```

### Evaluating
For evaluating, run the following command:

```
python3 main.py --root <path-to-image> --evaluation --model-to-load <path-to-model-pt> --amps-to-load <path-to-amp-pt> --num-steps <number-of-samples> --batch-size <number-of-images-in-batch>
```
