import torch,torch.nn as nn,torch.optim as optim,torchvision,matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.utils.data import DataLoader,Subset

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:",device)

tf=T.Compose([T.Resize((224,224)),T.ToTensor(),T.Normalize((.5,.5,.5),(.5,.5,.5))])

train_full=torchvision.datasets.CIFAR10("./data",train=True,download=True,transform=tf)
test=torchvision.datasets.CIFAR10("./data",train=False,download=True,transform=tf)

train=Subset(train_full,range(3000))
train_loader=DataLoader(train,batch_size=64,shuffle=True)
test_loader=DataLoader(test,batch_size=2,shuffle=True)

classes=train_full.classes

# ---------- Standard VGG Implementation ----------
class StandardVGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,64,3,padding=1),nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1),nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1),nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128,256,3,padding=1),nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1),nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*28*28,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )

    def forward(self,x):
        return self.classifier(self.features(x))

std_vgg=StandardVGG().to(device)

# ---------- Load pretrained VGG variants ----------
def load_vgg(name):
    model=getattr(torchvision.models,name)(weights="IMAGENET1K_V1")
    for p in model.features.parameters(): p.requires_grad=False
    model.classifier[6]=nn.Linear(4096,10)
    return model.to(device)

models={
"StandardVGG":std_vgg,
"VGG11":load_vgg("vgg11"),
"VGG13":load_vgg("vgg13"),
"VGG16":load_vgg("vgg16"),
"VGG19":load_vgg("vgg19")
}

# ---------- Train classifier heads ----------
loss_fn=nn.CrossEntropyLoss()

for name,model in models.items():
    optimizer=optim.Adam(model.parameters(),lr=.001)
    model.train()
    for x,y in train_loader:
        x,y=x.to(device),y.to(device)
        optimizer.zero_grad()
        loss=loss_fn(model(x),y)
        loss.backward()
        optimizer.step()
        break   # single batch training

# ---------- Show two images ----------
imgs,labels=next(iter(test_loader))
fig,ax=plt.subplots(1,2)

for i in range(2):
    ax[i].imshow(imgs[i].permute(1,2,0)*0.5+0.5)
    ax[i].set_title("True: "+classes[labels[i]])
    ax[i].axis("off")

plt.show()

# ---------- Predictions ----------
imgs=imgs.to(device)
print("\nPredictions from all VGG architectures:\n")

for name,model in models.items():
    model.eval()
    with torch.no_grad():
        preds=model(imgs).argmax(1).cpu()

    print(name,":",[classes[p] for p in preds])