import torch
import torch.nn as nn
from src.data import get_data_loaders

# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        self.model = nn.Sequential(
            #layer1
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3, padding = 1), #64x224x224
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout),

            #layer2
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3, padding = 1),#64x224x224
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(2,2),#64x112x112

            #layer3
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3, padding = 1),#128x112x112
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout),


            #layer4
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3, padding = 1), #128x112x112
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(2,2), #128x56x56

            #layer5
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3, padding = 1), #256x56x56
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(dropout),

            #layer6
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3, padding = 1), #256x56x56
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(2,2), #256x28x28

            #layer7
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3, padding = 1), #512x28x28
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(dropout),


            #layer8
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding = 1), #512x28x28
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(2,2), #512x7x7


            # #flatten using GAP and flatten methods
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            nn.Linear(512,256),
            nn.Dropout(dropout),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256,128),
            nn.Dropout(dropout),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # # nn.Linear(1500,500),
            # # nn.Dropout(dropout),
            # # nn.BatchNorm1d(500),
            # # nn.ReLU(),

            nn.Linear(128,num_classes)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        return self.model(x)


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
