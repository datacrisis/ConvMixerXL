# Augmentations and Activations: Pushing The Performance Of ConvNets

This project is built based on [[1]](https://github.com/locuslab/convmixer-cifar10) and [[2]](https://github.com/locuslab/convmixer).
## Run
```
!python train.py --lr-max=0.05 --ra-n=2 --ra-m=12 --wd=0.0 --scale=1.0 --jitter=0 --reprob=0 --epochs=25 --batch-size=256 --conv-ks=8 --psize=1 --depth=16
```

## References
[1] 
```
@misc{github, 
  author={Asher Trockman and Zico Kolter}, 
  title={convmixer-cifar10}, 
  year={2022}, 
  url={https://github.com/locuslab/convmixer-cifar10}, 
}
```

[2] 
```
@misc{github, 
  author={Asher Trockman and Zico Kolter}, 
  title={Patches Are All You Need?}, 
  year={2022}, 
  url={https://github.com/locuslab/convmixer}, 
}
```

[3]
```
@misc{github, 
  author={hysts}, 
  title={pytorch_cutmix}, 
  year={2019}, 
  url={https://github.com/hysts/pytorch_cutmix}, 
}
```

[4]
```
@misc{github, 
  author={hysts}, 
  title={pytorch_mixup}, 
  year={2018}, 
  url={https://github.com/hysts/pytorch_mixup}, 
}
```
