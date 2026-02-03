# from torchvision import transforms

# train_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

# val_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader

# TRAIN_DIR = r"D:\fypdataset\muzzle_dataset\train"
# VAL_DIR   = r"D:\fypdataset\muzzle_dataset\val1"

# # Load train first
# train_dataset = ImageFolder(TRAIN_DIR, transform=train_transform)
# class_to_idx = train_dataset.class_to_idx
# classes = train_dataset.classes

# # Load val
# val_dataset = ImageFolder(VAL_DIR, transform=val_transform)

# # 🔑 FORCE same class mapping
# val_dataset.class_to_idx = class_to_idx
# val_dataset.classes = classes

# # 🔑 FIX LABELS
# val_dataset.samples = [
#     (path, class_to_idx[path.split("\\")[-2]])
#     for path, _ in val_dataset.samples
# ]

# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False)

# num_classes = len(classes)
# print("Number of cows:", num_classes)
# import torch
# import torch.nn as nn
# import torchvision.models as models

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# teacher = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# teacher.fc = nn.Linear(teacher.fc.in_features, num_classes)
# teacher.to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(teacher.parameters(), lr=1e-3)
# def train_teacher(model, loader, epochs=50):
#     model.train()
#     for epoch in range(epochs):
#         correct, total, loss_sum = 0, 0, 0

#         for images, labels in loader:
#             images, labels = images.to(device), labels.to(device)

#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             loss_sum += loss.item()
#             preds = outputs.argmax(1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)

#         print(
#             f"Epoch [{epoch+1}/{epochs}] "
#             f"Loss: {loss_sum:.4f} "
#             f"Train Acc: {100*correct/total:.2f}%"
#         )

# train_teacher(teacher, train_loader)
# def evaluate(model, loader):
#     model.eval()
#     correct, total = 0, 0

#     with torch.no_grad():
#         for images, labels in loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             preds = outputs.argmax(1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)

#     print(f"Validation Accuracy: {100*correct/total:.2f}%")

# evaluate(teacher, val_loader)

#-------------------------------------------------------------------------------------------------------------------
import os
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

# Paths
TRAIN_DIR = r"D:\fypdataset\muzzle_dataset\train"
VAL_DIR = r"D:\fypdataset\muzzle_dataset\val1"

# Image transformations (ViT-friendly)
vit_mean = (0.5, 0.5, 0.5)
vit_std = (0.5, 0.5, 0.5)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(vit_mean, vit_std),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(vit_mean, vit_std),
])

# Load datasets
train_dataset = ImageFolder(root=TRAIN_DIR, transform=train_transform)
val_dataset = ImageFolder(root=VAL_DIR, transform=val_transform)

# Keep class mapping consistent between train/val
val_dataset.class_to_idx = train_dataset.class_to_idx
val_dataset.classes = train_dataset.classes
val_dataset.samples = [
    (path, train_dataset.class_to_idx[os.path.normpath(path).split(os.sep)[-2]])
    for path, _ in val_dataset.samples
]

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

# Info
num_classes = len(train_dataset.classes)
print("Classes (Cow IDs):", train_dataset.classes)
print("Number of cows:", num_classes)
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN Teacher (ResNet18)
teacher = models.resnet18(pretrained=True)
teacher.fc = nn.Linear(teacher.fc.in_features, num_classes)
teacher = teacher.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(teacher.parameters(), lr=1e-4, weight_decay=0.01)
def train_teacher(model, loader, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Teacher Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.4f}")

train_teacher(teacher, train_loader)
torch.save(teacher.state_dict(), "teacher_cnn.pth")
import timm

student = timm.create_model(
    "vit_base_patch16_224",
    pretrained=True,
    num_classes=num_classes
).to(device)

student_optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=0.05)
student_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(student_optimizer, T_max=15)

def distillation_loss(student_logits, teacher_logits, labels, alpha=0.7, T=4):
    hard_loss = criterion(student_logits, labels)
    soft_loss = nn.KLDivLoss(reduction="batchmean")(
        torch.log_softmax(student_logits / T, dim=1),
        torch.softmax(teacher_logits / T, dim=1)
    )
    return alpha * hard_loss + (1 - alpha) * (T * T) * soft_loss
def train_student(student, teacher, loader, epochs=15):
    teacher.eval()
    student.train()

    for epoch in range(epochs):
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                teacher_outputs = teacher(images)

            student_outputs = student(images)
            loss = distillation_loss(student_outputs, teacher_outputs, labels)

            student_optimizer.zero_grad()
            loss.backward()
            student_optimizer.step()

            total_loss += loss.item()

        student_scheduler.step()
        print(f"Student Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.4f}")

train_student(student, teacher, train_loader)
torch.save(student.state_dict(), "student_vit.pth")
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
#print(f"Total images found in train: {len(train_filenames)}")
#print("Train file names sample:", train_filenames[:5])
#print("Labels extracted:", labels_raw[:5])

    #print("Validation Accuracy is: 97.01777")
    #print("Unique label mapping:", label_to_idx)
    #print("Total classes:", len(label_to_idx))
    print(f"Accuracy: {acc:.2f}%")

evaluate(student, val_loader)

