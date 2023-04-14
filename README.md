<br />
<p align="center">

  <h1 align="center">Activations And Augmentations: Pushing The Performance Of Isotropic ConvNets</h1>
  <h4 align="center"><a href="https://github.com/datacrisis">Keifer Lee</a>, <a href="https://github.com/iamshubhamgupto">Shubham Gupta</a>, <a href="">Karan Sharma</a></h4>
  
</p>

[Colab Demo](https://colab.research.google.com/drive/13harjTZPcYIJJ5UQgUSMK6m8QpLQ7Rcj?usp=sharing)

# Abstract

<p align="justify">
   Isotropic architectures have recently gained focus for solving computer vision problems for their ability to capture better spatial information. In this work, we experiment with training a ConvMixer model, an isotropic convolutional neural net architecture on the CIFAR-10 dataset. We propose a new architecture: ConvMixer-XL consisting of 66 layers and just under $5M$ parameters. To maximize its performance, various configurations of the architecture, augmentations and activations were tried in our ablation study to further fine-tune the model. Our experiments show applying augmentations and using the Swish (SiLU) activation function for deeper models gives the best results with a top-1 accuracy of 94.52%. Our code can be found at   <a href="https://github.com/datacrisis/ConvMixerXL">https://github.com/datacrisis/ConvMixerXL</a>.
</p>

# Results

| **Name**      | **Activation** | **Depth** | **Inter-Block Skip** | **Augmentations** | **#Params (M)** | **Top 1 %Acc** |
|------------------|------------------------------|-------------------------|------------------------------------|---------------------------------|--------------------------------|-------------------------------|
| CM-Vanilla-NoAug | GELU                         | 8                       | No                                 | No                              | 0.59                           | 0.8854                        |
| CM-Vanilla       | GELU                         | 8                       | No                                 | Yes                             | 0.59                           | 0.9378                        |
| CM-Vanilla-ReLU  | ReLU                         | 8                       | No                                 | Yes                             | 0.59                           | 0.9384                        |
| CM-Vanilla-SiLU  | SiLU                         | 8                       | No                                 | Yes                             | 0.59                           | 0.9372                        |
| CM-XL-NoSkip     | GELU                         | 66                      | No                                 | Yes                             | 4.9                            | 0.4868                        |
| CM-XL-Skip       | GELU                         | 66                      | Yes                                | Yes                             | 4.9                            | 0.9422                        |
| **CM-XL**   | **SiLU**                | **66**             | **Yes**                       | **Yes**                    | **4.9**                   | **0.9452**               |

## Weights and logs
We have uploaded all our experimentation logging and weights generated [here](https://drive.google.com/file/d/1DYqkYPayh6tKxsz5TX3--15t1twtqdKJ).

# Run
To reproduce the best performing configuration of ConvMixerXL, run the following code:
```shell
python3 train.py --lr-max=0.005 \
                  --depth=66\
                  --model='CM-XL'\
                  --activation='SiLU'\
                  --name='final_CMXL_SiLU'\
                  --save_dir='output/agg'\
                  --batch-size=128
```

## References

This project is built based on [ConvMixer CIFAR-10](https://github.com/locuslab/convmixer-cifar10) and [ConvMixer](https://github.com/locuslab/convmixer).

