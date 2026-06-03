import torch,torch.nn as nn,torch.optim as optim,re,matplotlib.pyplot as plt

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:",device)

text="""alice was beginning to get very tired of sitting by her sister on the bank
and of having nothing to do once or twice she had peeped into the book her sister was reading
but it had no pictures or conversations in it and what is the use of a book thought alice"""

tokens=re.findall(r'\b\w+\b',text.lower())
vocab=sorted(set(tokens))
w2i={w:i for i,w in enumerate(vocab)}
i2w={i:w for w,i in w2i.items()}

window=3
X,y=[],[]

for i in range(len(tokens)-window):
    X.append([w2i[w] for w in tokens[i:i+window]])
    y.append(w2i[tokens[i+window]])

X=torch.tensor(X).to(device)
y=torch.tensor(y).to(device)

class CNN1D(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.embed=nn.Embedding(vocab_size,32)
        self.conv=nn.Conv1d(32,64,kernel_size=window)
        self.fc=nn.Linear(64,vocab_size)
    def forward(self,x):
        x=self.embed(x).permute(0,2,1)
        x=torch.relu(self.conv(x)).squeeze(-1)
        return self.fc(x)

model=CNN1D(len(vocab)).to(device)

loss_fn=nn.CrossEntropyLoss()
opt=optim.Adam(model.parameters(),lr=.01)

losses=[]

for epoch in range(200):
    opt.zero_grad()
    pred=model(X)
    loss=loss_fn(pred,y)
    loss.backward()
    opt.step()
    losses.append(loss.item())

print("Final Loss:",losses[-1])

plt.plot(losses);plt.title("1D CNN Training Loss");plt.show()

print("\nCNN Kernel Size (window size):",window)
print("Conv Layer Weight Shape:",model.conv.weight.shape)

def predict(words):
    seq=torch.tensor([[w2i[w] for w in words]]).to(device)
    with torch.no_grad():
        p=model(seq).argmax(1).item()
    return i2w[p]

tests=[
["alice","was","beginning"],
["she","had","peeped"],
["the","use","of"]
]

print("\nSliding Window Predictions Using 1D CNN:\n")

for t in tests:
    print("Input Window:",' '.join(t),"-> Predicted Next Word:",predict(t))