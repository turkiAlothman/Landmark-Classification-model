import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()
        self.model = nn.Sequential(
        
        # the backbone (convolutional layers + maxPool)

            
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),# -> 16x224x224
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),# -> 16
            
            nn.Conv2d(16, 32, 3, padding=1),  # -> 32x112x112
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 32x56x56
            
            nn.Conv2d(32, 64, 3, padding=1),  # -> 64x56x56
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 64
            
            
            nn.Conv2d(64, 128, 3, padding=1),  # -> 128x28x28
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 128x14x14
            
 
            
            

        
            nn.Flatten(),
            
            
            # the head(fully connected layer) 
            
            nn.Linear(128 * 14 *14, 500),  # -> 500
            nn.Dropout(0.5),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, num_classes),
        
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
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
