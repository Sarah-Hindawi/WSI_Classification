import time
import copy
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score, accuracy_score

import config

def train(train_loader, valid_loader, model, criterion, optimizer, scheduler, device):

    best_acc = 0.0
    best_model = None
    early_stopping_counter = 0

    num_epochs = 25
    for epoch in range(num_epochs):
        model.train()
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)

            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validate the model
        model.eval()
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                preds = (torch.sigmoid(outputs) > 0.5).float() 
                
                valid_acc = accuracy_score(preds.cpu().data.numpy(),labels.cpu().data.numpy())
                valid_f1_score = f1_score(preds.cpu().data.numpy(),labels.cpu().data.numpy())

        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {valid_acc:.2f}%, F1-score: {valid_f1_score:.2f}%")

        if valid_acc > best_acc:
            best_model = copy.deepcopy(model)
            best_acc = valid_acc

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, config.BEST_MODEL_VALIDATE_PATCHES_PATH)
            
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= 5:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    return best_model


def test(test_loader, model, device):

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float() 
            test_acc = accuracy_score(preds.cpu().data.numpy(),labels.cpu().data.numpy())
            test_f1_score = f1_score(preds.cpu().data.numpy(),labels.cpu().data.numpy())

    print(f"Test Accuracy: {test_acc:.2f}%, F1-score: {test_f1_score:.2f}%")


def infer(test_loader, model, device):
    pass

def train_model():

    dataset = datasets.ImageFolder(root=config.VALIDATE_PATCHES_PATH, transform=transforms.ToTensor()) # Patches in dataset are already normalized

    num_samples = len(dataset)
    num_train = int(0.8 * num_samples)
    num_valid = int(0.1 * num_samples)
    num_test = num_samples - num_train - num_valid

    train_dataset, valid_dataset, test_dataset = random_split(dataset, [num_train, num_valid, num_test])

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(512, 1)  # Modify the last layer to only output the logit for the positive class
    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.7)

    start_time = time.time()
    best_model = train(train_loader, valid_loader, model, criterion, optimizer, scheduler, device)
    test(test_loader, best_model, device)

    print(f"Completed training the patch validation model in: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    train_model()