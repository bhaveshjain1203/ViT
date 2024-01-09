import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.optim as optim
from tqdm import tqdm # for progress bar

from utils import (
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
from model import vit
import matplotlib.pyplot as plt

TRAIN_IMG_DIR = "train_images_with_masks/images/"
TRAIN_MASK_DIR = "train_images_with_masks/masks/"
VAL_IMG_DIR = "val_images_with_masks/images/"
VAL_MASK_DIR = "val_images_with_masks/masks/"

LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
IMAGE_HEIGHT = 128  
IMAGE_WIDTH = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_losses = []
val_accuracies = []

def train(loader, model, optimizer, loss_fn, scaler):

    # Intialize progress bar
    loop = tqdm(loader)
    total_loss = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)

        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward pass
        # with torch.cuda.amp.autocast(): # ??? TODO 
        predictions = model(data)
        # print(predictions.shape)
        # print(targets.shape)
        loss = loss_fn(predictions, targets)

        # backward pass

        # clear the gradients
        optimizer.zero_grad()
        loss.backward() # backpropagation
        optimizer.step() # update weights

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    
    avg_loss = total_loss / len(loader)
    train_losses.append(avg_loss)

def main():
    train_transform = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Rotate(limit=35, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
        )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    train_loader, val_loader = get_loaders(
            TRAIN_IMG_DIR,
            TRAIN_MASK_DIR,
            VAL_IMG_DIR,
            VAL_MASK_DIR,
            train_transform,
            val_transforms
        )

    (train_pets_inputs, train_pets_targets) = next(iter(train_loader))
    (test_pets_inputs, test_pets_targets) = next(iter(val_loader))
    print("input shape -> ",train_pets_inputs.shape)
    print("output shape ->",train_pets_targets.shape)

    model = vit.to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss() # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    scaler = torch.cuda.amp.GradScaler()

    print("train")
    for epoch in range(NUM_EPOCHS):
        print("epoch:",epoch)
        train(train_loader, model, optimizer, loss_fn, scaler)
        print("training done")

        # check accuracy
        val_accuracy = check_accuracy(val_loader, model, device=DEVICE)
        val_accuracies.append(val_accuracy)

        print("accuracy checked")

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )
        print("saved predictions")
    plot_graphs(train_losses, val_accuracies)

def plot_graphs(train_losses, val_accuracies):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print('hello')
    main()



