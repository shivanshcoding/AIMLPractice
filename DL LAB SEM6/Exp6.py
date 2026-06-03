import torch,torch.nn as nn,torch.optim as optim,re,matplotlib.pyplot as plt
from collections import Counter

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:",device)

# small english corpus
text="""alice was beginning to get very tired of sitting by her sister on the bank
and of having nothing to do once or twice she had peeped into the book her sister was reading
but it had no pictures or conversations in it and what is the use of a book thought alice
without pictures or conversation"""

tokens=re.findall(r'\b\w+\b',text.lower())

# vocabulary
vocab=sorted(set(tokens))
word2idx={w:i for i,w in enumerate(vocab)}
idx2word={i:w for w,i in word2idx.items()}

# sequences
seq_len=3
X,y=[],[]

for i in range(len(tokens)-seq_len):
    X.append([word2idx[w] for w in tokens[i:i+seq_len]])
    y.append(word2idx[tokens[i+seq_len]])

X=torch.tensor(X).to(device)
y=torch.tensor(y).to(device)

# model
class RNN(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.embed=nn.Embedding(vocab_size,32)
        self.rnn=nn.RNN(32,64,batch_first=True)
        self.fc=nn.Linear(64,vocab_size)
    def forward(self,x):
        x=self.embed(x)
        out,_=self.rnn(x)
        return self.fc(out[:,-1,:])

model=RNN(len(vocab)).to(device)

loss_fn=nn.CrossEntropyLoss()
opt=optim.Adam(model.parameters(),lr=.01)

# training
losses=[]
for epoch in range(200):
    opt.zero_grad()
    pred=model(X)
    loss=loss_fn(pred,y)
    loss.backward()
    opt.step()
    losses.append(loss.item())

print("Final Loss:",losses[-1])

plt.plot(losses)
plt.title("RNN Training Loss")
plt.show()

# inference
def predict(words):
    model.eval()
    seq=torch.tensor([[word2idx[w] for w in words]]).to(device)
    with torch.no_grad():
        pred=model(seq).argmax(1).item()
    return idx2word[pred]

print("\nNext Word Predictions:\n")

tests=[
["alice","was","beginning"],
["she","had","peeped"],
["the","use","of"]
]

for t in tests:
    print("Input:",' '.join(t),"-> Predicted:",predict(t))