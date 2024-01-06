import config
import time
import copy
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score


def train(train_loader, valid_loader, model, num_epochs, criterion, optimizer, scheduler, device):

    best_epoch = None
    best_model = model
    best_metric = 0.0
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        model.train()
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            
            optimizer.zero_grad()

            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
        
        scheduler.step()

        train_acc, train_roc_auc, train_f1_score = evaluate(train_loader, model, device)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Accuracy: {train_acc:.2f}%, ROC AUC: {train_roc_auc:.2f}%, F1-score: {train_f1_score:.2f}%")

        # Validate the model
        if valid_loader:
            valid_acc, valid_roc_auc, valid_f1_score = evaluate(valid_loader, model, device)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {valid_acc:.2f}%, ROC AUC: {valid_roc_auc:.2f}%, F1-score: {valid_f1_score:.2f}%")

            if valid_roc_auc > best_metric:
                best_epoch = epoch
                best_model = copy.deepcopy(model)
                best_metric = valid_roc_auc
                
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= 5:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    return best_model, best_epoch


def evaluate(loader, model, device):

    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float() 

            all_preds.append(preds.cpu().data.numpy())
            all_labels.append(labels.cpu().data.numpy())
                
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        test_acc = accuracy_score(all_labels, all_preds)
        test_roc_auc = roc_auc_score(all_labels, all_preds)
        test_f1_score = f1_score(all_labels, all_preds)

    return test_acc, test_roc_auc, test_f1_score

def generate_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(root=config.VALIDATE_PATCHES_PATH, transform=transform)
    
    targets = [item[1] for item in dataset.samples] 

    train_indices, valid_test_indices, _, _ = train_test_split(range(len(targets)), targets, stratify=targets, test_size=0.3, random_state=42)    
    valid_indices, test_indices = train_test_split(valid_test_indices, stratify=[targets[i] for i in valid_test_indices], test_size=0.5, random_state=42)

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    valid_subset = torch.utils.data.Subset(dataset, valid_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)

    batch_size = 32
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


def create_model(train_loader, valid_loader, test_loader, model,criterion, optimizer, scheduler, num_epochs, device):
    start_time = time.time()
    print(f"Started training...")

    best_model, _ = train(train_loader, valid_loader, model, num_epochs=num_epochs, criterion=criterion, optimizer=optimizer, scheduler=scheduler, device=device)
    
    torch.save({
        'model_state_dict': best_model.state_dict(),
        }, config.BEST_MODEL_VALIDATE_PATCHES_PATH)

    test_acc, test_roc_auc, test_f1_score = evaluate(test_loader, best_model, device)
    print(f"Test Accuracy: {test_acc*100:.3f}%, ROC AUC: {test_roc_auc:.4f}, F1-score: {test_f1_score:.4f}")

    print(f"Completed training and testing in: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(512, 1)  # Only output for the positive class
    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=4, gamma=0.7)

    train_loader, valid_loader, test_loader = generate_dataloaders()
    create_model(train_loader, valid_loader, test_loader, model, criterion, optimizer, scheduler, num_epochs=30, device=device)