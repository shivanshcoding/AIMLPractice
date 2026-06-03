import torch,torch.nn as nn,torch.optim as optim,torchvision,matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.utils.data import DataLoader

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:",device)

tf=T.Compose([T.ToTensor(),T.Normalize((.5,.5,.5),(.5,.5,.5))])

train=torchvision.datasets.CIFAR10("./data",train=True,download=True,transform=tf)
test=torchvision.datasets.CIFAR10("./data",train=False,download=True,transform=tf)

train_loader=DataLoader(train,batch_size=128,shuffle=True)
test_loader=DataLoader(test,batch_size=128)

classes=train.classes

class DeepCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64*8*8,256),
            nn.ReLU(),
            nn.Linear(256,10)
        )

    def forward(self,x): return self.net(x)

model=DeepCNN().to(device)

print("\nPadding Used: padding=1 in convolution layers")
print("BatchNorm Layers:",[m for m in model.modules() if isinstance(m,nn.BatchNorm2d)])

loss_fn=nn.CrossEntropyLoss()
opt=optim.Adam(model.parameters(),lr=.001)

losses=[]

for epoch in range(5):
    total=0
    for x,y in train_loader:
        x,y=x.to(device),y.to(device)

        opt.zero_grad()
        loss=loss_fn(model(x),y)
        loss.backward()
        opt.step()

        total+=loss.item()

    losses.append(total/len(train_loader))
    print(f"Epoch {epoch+1} Loss:{losses[-1]:.4f}")

plt.plot(losses);plt.title("Training Loss Curve");plt.show()

model.eval()

imgs,labels=next(iter(test_loader))
imgs_show=imgs[:2]

fig,ax=plt.subplots(1,2)

for i in range(2):
    ax[i].imshow(imgs_show[i].permute(1,2,0)*0.5+0.5)
    ax[i].set_title("True: "+classes[labels[i]])
    ax[i].axis("off")

plt.show()

with torch.no_grad():
    preds=model(imgs_show.to(device)).argmax(1).cpu()

print("\nPredictions on Test Images:\n")

for i,p in enumerate(preds):
    print(f"Image {i+1} Predicted:",classes[p])