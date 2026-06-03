import torch,torch.nn as nn,torch.optim as optim,matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:",device)

vocab_size,max_len=10000,200
(X_train,y_train),(X_test,y_test)=imdb.load_data(num_words=vocab_size)

X_train=pad_sequences(X_train,maxlen=max_len)
X_test=pad_sequences(X_test,maxlen=max_len)

X_train=torch.tensor(X_train).to(device)
y_train=torch.tensor(y_train).to(device)
X_test=torch.tensor(X_test).to(device)
y_test=torch.tensor(y_test).to(device)

class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.e=nn.Embedding(vocab_size,64)
        self.r=nn.RNN(64,128,batch_first=True)
        self.f=nn.Linear(128,2)
    def forward(self,x):
        x=self.e(x)
        o,_=self.r(x)
        return self.f(o[:,-1,:])

model=RNN().to(device)
loss_fn=nn.CrossEntropyLoss()
opt=optim.Adam(model.parameters(),lr=.001)

losses=[]
for epoch in range(4):
    idx=torch.randperm(len(X_train))[:4000]
    opt.zero_grad()
    pred=model(X_train[idx])
    loss=loss_fn(pred,y_train[idx])
    loss.backward()
    opt.step()
    losses.append(loss.item())
    print(f"Epoch {epoch+1} Loss:{loss.item():.4f}")

plt.plot(losses);plt.title("Training Loss");plt.show()

model.eval()
with torch.no_grad():
    acc=(model(X_test[:2000]).argmax(1)==y_test[:2000]).float().mean()*100

print(f"\nTest Accuracy:{acc:.2f}%")

word_index=imdb.get_word_index()
rev_index={v:k for k,v in word_index.items()}

def decode(x): return " ".join([rev_index.get(i-3,"?") for i in x if i>3][:25])

pos,neg=False,False
i=0

print("\nSentiment Predictions:\n")

while not(pos and neg):
    sample=X_test[i].unsqueeze(0)
    with torch.no_grad():
        p=model(sample).argmax(1).item()
    print(decode(sample.squeeze().cpu().numpy()),"...")
    print("Sentiment:",("Positive 🙂" if p==1 else "Negative 🙁"),"\n")
    if p==1: pos=True
    else: neg=True
    i+=1