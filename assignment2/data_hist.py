import torch
from torch . utils . data import DataLoader
from torchtext import datasets
from torch . utils . data . backward_compatibility import worker_init_fn
import matplotlib.pyplot as plt
from collections import defaultdict

train_data = datasets . UDPOS (split ='train')

def pad_collate ( batch ):
    xx = [b [0] for b in batch ]
    yy = [b [1] for b in batch ]

    x_lens = [ len(x) for x in xx]

    return xx , yy , x_lens

train_loader = DataLoader ( dataset = train_data , batch_size =50,
shuffle =True , num_workers =1,
worker_init_fn = worker_init_fn ,
drop_last =True , collate_fn = pad_collate)

xx ,yy , xlens = next ( iter ( train_loader ))

print(xx)

def visualizeSentenceWithTags (text , udtags ):
    print (" Token "+"". join ([" " ]*(15) )+" POS Tag ")
    print (" ---------------------------------")
    for w, t in zip (text , udtags ):
        print (w+"". join ([" " ]*(20 - len (w)))+t)

visualizeSentenceWithTags (xx[0] , yy[0])

freq = defaultdict(int)

for _, tags, _ in train_data:
    for tag in tags:
        freq[tag] += 1


x = []
y = []

for key in freq:
    x.append(key)
    y.append(freq[key])

plt.figure(figsize=(14, 7))
plt.bar(x, y)

plt.xlabel('Tags')
plt.ylabel('Frequency')
plt.title('Frequency of Tags')

plt.show()
plt.savefig("histogram.png")