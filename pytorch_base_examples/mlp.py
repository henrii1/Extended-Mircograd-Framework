import torch
import requests

# data_link = 'https://raw.githubusercontent.com/karpathy/makemore/master/names.txt'
# file = requests.get(data_link)
# with open('names.txt', 'wb') as f:
#   f.write(file.content)

# words = open('names.txt', 'r').read().splitlines()

#alternative approach
!wget https://raw.githubusercontent.com/karpathy/makemore/master/names.txt

words = open('names.txt', 'r').read().splitlines()

# comment this part in the repo
# word_count = len(words)
# max_length = max(len(word) for word in words)
# min_length = min(len(word) for word in words)

# print(f"{word_count=} \n {max_length=} \n {min_length=}")


# create a lookup table for strings
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}


#creating dataset
xx, ys = [], []

for word in words:
  chs = ['.'] + list(word) + ['.']
  for x, y in zip(chs, chs[1:]):
    ix = stoi[x]
    iy = stoi[y]

    xx.append(ix)
    ys.append(iy)

xs = torch.tensor(xx)
ys = torch.tensor(ys)

num = xs.nelement()



# use a generator since we aren't using nn.Module
g = torch.Generator().manual_seed(23485763543) # for reproduciblity
W = torch.randn(27, 27, generator=g, requires_grad=True)


# we are not sampling, we are going to one hot everything
def one_hot(lst: list):
    zeros = torch.zeros(len(lst), 27).float()
    for i, v in enumerate(lst):
        zeros[i, v-1] = 1
    #zeros.requires_grad_(True)
    return zeros

# training for 10 steps without using nn.Functional
lr = 50
xenc = one_hot(xx)
for k in range(10):

  #forward pass
  #xenc = one_hot(xx) || F.one_hot(xs, num_classes=27).float()
  logits = xenc @ W   #without bias

  #softmax function
  counts = logits.exp()
  probs = counts / counts.sum(1, keepdim=True)

  #crossentropy calculation and (L2 Norm)
  #L2 Norm: scale the mean of the weight square
  loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
  print(loss.item())

  #gradient computation
  W.grad = None     # set gradient to zero
  loss.backward()

  #update data using SGD
  W.data += -lr * W.grad 

#L2 Norm is applied to weight vectors, it ensures that their magnitudes do not vary widely.


# Sampling from model
# Sampling from the model
g = torch.Generator().manual_seed(234578373)

for i in range(5):

  out = []
  ix = 0
  while True:
    xenc = one_hot([ix]) #F.one_hot(torch.tensor([ix]), num_classes= 27).float()  # shape 1, 27
    logits = xenc @ W   # shape 1, 27
    counts = logits.exp()
    p = counts / counts.sum(1, keepdim = True)

    ix = torch.multinomial(p, num_samples= 1, replacement=True, generator=g).item()  # single value based on p distribution

    out.append(itos[ix])
    if ix == 0:
      break
  print(''.join(out))
