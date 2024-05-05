import torch
from torch . utils . data import DataLoader
from torchtext import datasets
from torch . utils . data . backward_compatibility import worker_init_fn
import matplotlib.pyplot as plt
from collections import defaultdict

# Create data pipeline
train_data = datasets . UDPOS (split ='train')

# Function to combine data elements from a batch
def pad_collate ( batch ):
    xx = [b [0] for b in batch ]
    yy = [b [1] for b in batch ]

    x_lens = [ len(x) for x in xx]

    return xx , yy , x_lens

 # Make data loader
train_loader = DataLoader ( dataset = train_data , batch_size =50,
shuffle =True , num_workers =1,
worker_init_fn = worker_init_fn ,
drop_last =True , collate_fn = pad_collate)

# Look at the first batch
xx ,yy , xlens = next ( iter ( train_loader ))

print(xx)

# Visualizing POS tagged sentence
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

# Plotting the histogram
plt.figure(figsize=(14, 7))
plt.bar(x, y, color='green')

# Adding labels and title
plt.xlabel('POS Tags')
plt.ylabel('Frequency')
plt.title('Frequency of POS Tags')

# Rotating the x-axis labels for better readability
plt.xticks(rotation=0)

# Display the plot
plt.tight_layout()
plt.show()