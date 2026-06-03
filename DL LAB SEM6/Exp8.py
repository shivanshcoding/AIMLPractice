import torch,torch.nn as nn,torch.optim as optim,re,matplotlib.pyplot as plt

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:",device)

text="""alice was beginning to get very tired of sitting by her sister on the bank
and of having nothing to do once or twice she had peeped into the book her sister was reading
but it had no pictures or conversations in it and what is the use of a book thought alice
without pictures or conversation so she was considering in her own mind"""

tokens=re.findall(r'\b\w+\b',text.lower())
vocab=sorted(set(tokens))
w2i={w:i for i,w in enumerate(vocab)}
i2w={i:w for w,i in w2i.items()}

seq_len=3
X,y=[],[]

for i in range(len(tokens)-seq_len):
    X.append([w2i[w] for w in tokens[i:i+seq_len]])
    y.append(w2i[tokens[i+seq_len]])

X=torch.tensor(X).to(device)
y=torch.tensor(y).to(device)

class LSTM(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.e=nn.Embedding(vocab_size,32)
        self.l=nn.LSTM(32,64,batch_first=True)
        self.f=nn.Linear(64,vocab_size)
    def forward(self,x):
        x=self.e(x)
        o,_=self.l(x)
        return self.f(o[:,-1,:])

model=LSTM(len(vocab)).to(device)

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

plt.plot(losses);plt.title("LSTM Training Loss");plt.show()

def predict(words):
    seq=torch.tensor([[w2i[w] for w in words]]).to(device)
    with torch.no_grad():
        p=model(seq).argmax(1).item()
    return i2w[p]

tests=[
["alice","was","beginning"],
["she","had","peeped"],
["the","use","of"],
["her","sister","was"]
]

print("\nNext Word Predictions Using LSTM:\n")

for t in tests:
    print("Input:",' '.join(t),"-> Predicted:",predict(t))