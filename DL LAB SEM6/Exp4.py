import torch,torch.nn as nn,torch.optim as optim,matplotlib.pyplot as plt,re
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from collections import Counter

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:",device)

categories=["sci.space","rec.sport.hockey","comp.sys.ibm.pc.hardware","talk.politics.misc"]

data=fetch_20newsgroups(subset="train",categories=categories,
remove=("headers","footers","quotes"))

texts=data.data[:4000]
labels=data.target[:4000]

label_map={
0:"Sci/Tech",
1:"Sports",
2:"Business",
3:"World"
}

def tokenize(t): return re.findall(r'\b\w+\b',t.lower())

counter=Counter()
for t in texts: counter.update(tokenize(t))

vocab={w:i+1 for i,(w,_) in enumerate(counter.most_common(12000))}

max_len=120

def encode(text):
    tokens=[vocab.get(w,0) for w in tokenize(text)]
    return tokens[:max_len]+[0]*(max_len-len(tokens[:max_len]))

X=torch.tensor([encode(t) for t in texts])
y=torch.tensor(labels)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)

X_train,X_test=X_train.to(device),X_test.to(device)
y_train,y_test=y_train.to(device),y_test.to(device)

class TextCNN(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.embed=nn.Embedding(vocab_size,128)
        self.conv=nn.Conv1d(128,128,5)
        self.pool=nn.AdaptiveMaxPool1d(1)
        self.fc=nn.Linear(128,4)
    def forward(self,x):
        x=self.embed(x).permute(0,2,1)
        x=torch.relu(self.conv(x))
        x=self.pool(x).squeeze(-1)
        return self.fc(x)

model=TextCNN(len(vocab)+1).to(device)

loss_fn=nn.CrossEntropyLoss()
opt=optim.Adam(model.parameters(),lr=.001)

losses=[]

for epoch in range(5):
    opt.zero_grad()
    pred=model(X_train)
    loss=loss_fn(pred,y_train)
    loss.backward()
    opt.step()
    losses.append(loss.item())
    print(f"Epoch {epoch+1} Loss:{loss.item():.4f}")

plt.plot(losses);plt.title("CNN Training Loss");plt.show()

model.eval()

with torch.no_grad():
    preds=model(X_test).argmax(1)
    acc=(preds==y_test).float().mean()*100

print(f"\nTest Accuracy:{acc:.2f}%")

print("\nCNN Kernel Size:",model.conv.kernel_size)

print("\nSentence-Level Actual vs Predicted:\n")

for i in range(3):
    i+=3
    sentence=texts[i][:120]
    actual=label_map[y[i].item()]
    predicted=label_map[preds[i].item()]

    print("Sentence:")
    print(sentence," ... so on")
    print("Actual:",actual)
    print("Predicted:",predicted,"\n")