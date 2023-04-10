import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def ConvMixer(dim, depth, kernel_size=5, patch_size=2, n_classes=10, activation='GeLU'):
    
    #Det activation func
    if activation == 'GELU':
        act_fx = nn.GELU()
    elif activation == 'ReLU':
        act_fx = nn.ReLU()
    elif activation == 'SiLU':
        act_fx = nn.SiLU()
    elif activation == 'LeakyReLU':
        act_fx = nn.LeakyReLU()

    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        act_fx,
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    act_fx,
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                act_fx,
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )


def ConvMixerXL(dim, depth, kernel_size=5, patch_size=2, n_classes=10, skip_period=3, activation='GeLU'):

    #Det activation func
    if activation == 'GELU':
        act_fx = nn.GELU()
    elif activation == 'ReLU':
        act_fx = nn.ReLU()
    elif activation == 'SiLU':
        act_fx = nn.SiLU()
    elif activation == 'LeakyReLU':
        act_fx = nn.LeakyReLU()

    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        act_fx,
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                        Residual(nn.Sequential(*[nn.Sequential(
                                                Residual(
                                                        nn.Sequential(
                                                                        nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                                                                        act_fx,
                                                                        nn.BatchNorm2d(dim)
                                                                        )
                                                        ),
                                                nn.Conv2d(dim, dim, kernel_size=1),
                                                act_fx,
                                                nn.BatchNorm2d(dim)
                                                ) for i in range(depth//skip_period)] 
                                              )
                                ) 
                        ) for i in range(skip_period)
        ],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )