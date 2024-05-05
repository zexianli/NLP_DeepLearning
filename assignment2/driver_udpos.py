import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torchtext.datasets import UDPOS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch.nn.functional as F
from tqdm import tqdm
import nltk
import matplotlib.pyplot as plt
from nltk import word_tokenize
nltk.download('punkt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, tagset_size):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim//2, num_layers=1, bidirectional=True)
        self.fc = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed_embedded)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        tag_space = self.fc(out)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
    

def train_model(model, train_loader, validation_loader, epochs=10, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()  
    training_losses = []
    validation_losses = []

    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0
        total_elements = 0

        model.train()
        for texts, tags, lengths in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            texts = texts.to(device)
            tags = tags.to(device)       

            optimizer.zero_grad() 
            out = model(texts, lengths) 

            out = out.view(-1, out.shape[-1]) 
            tags = tags.view(-1)  

            loss = criterion(out, tags) 
            loss.backward()  
            optimizer.step() 

            _, predicted = torch.max(out, 1)
            correct = (predicted == tags).float().sum()
            total_correct += correct.item()
            total_loss += loss.item() * texts.size(0)
            total_elements += tags.numel()

        test_loss = total_loss / total_elements
        accuracy = total_correct / total_elements
        print(f"Epoch {epoch + 1}: Training Loss: {test_loss:.2f}, Training Accuracy: {accuracy:.2f}")

        val_loss, val_accuracy = validation(model, validation_loader)
        print(f"Validation Loss: {val_loss:.2f}, Validation Accuracy: {val_accuracy:.2f}")

        training_losses.append(test_loss)
        validation_losses.append(val_loss)

    plt.plot([i for i in range(1, epochs+1)], training_losses, label='Training Loss', marker='o', color='green')
    plt.plot([i for i in range(1, epochs+1)], validation_losses, label='Validation Loss', marker='s', color='red')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()

    plt.show()
    plt.savefig("udpos.png")


def validation(model, val_loader):
    model.eval() 
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_elements = 0

    with torch.no_grad():  
        for texts, tags, lengths in val_loader:
            texts = texts.to(device)
            tags = tags.to(device)

            outputs = model(texts, lengths) 
            outputs = outputs.view(-1, outputs.shape[-1])
            tags = tags.view(-1)

            cur_loss = criterion(outputs, tags)  
            total_loss += cur_loss.item() * texts.size(0)

            _, predicted = torch.max(outputs, 1)
            correct = (predicted == tags).float().sum()
            total_correct += correct.item()
            total_elements += tags.numel()

    final_loss = total_loss / total_elements 
    accuracy = total_correct / total_elements  

    return final_loss, accuracy

def test(model, test_loader):
    model.eval() 
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_elements = 0

    with torch.no_grad():
        for texts, tags, lengths in test_loader:
            texts = texts.to(device)
            tags = tags.to(device)

            out = model(texts, lengths) 
            out = out.view(-1, out.shape[-1])
            tags = tags.view(-1)

            cur_loss = criterion(out, tags) 
            total_loss += cur_loss.item() * texts.size(0) 

            _, prediction = torch.max(out, 1)
            correct = (prediction == tags).float().sum()
            total_correct += correct.item()
            total_elements += tags.numel()

    final_loss = total_loss / total_elements
    accuracy = total_correct / total_elements

    return final_loss, accuracy

######################################################################
# Task 3.3
######################################################################
def tag_sentence(tk_vocabulary, tag_vocabulary, model, sentence):
    model.eval()
    tokens = word_tokenize(sentence)
    input_sentence = []

    for token in tokens:
        input_sentence.append(tk_vocabulary[token])

    input_tensor = torch.nn.utils.rnn.pad_sequence(torch.tensor([input_sentence]).to(device), padding_value=tk_vocabulary["<pad>"], batch_first=True)

    output = model(input_tensor, torch.tensor([len(tokens)]))
    output_tensor = torch.max(output, dim=2)[1].flatten()
    output_string = []

    for element in output_tensor:
        output_string.append(tag_vocabulary.get_itos()[element.item()])

    print(tokens)
    print(output_string)

def main():
    train_dataset, val_dataset, test_dataset = UDPOS()
    
    tokenizer = get_tokenizer('basic_english')
    vocabulary_tks = build_vocab_from_iterator(map(tokenizer, (word for entry in train_dataset for word in entry[0])), specials=["<unk>", "<pad>"])
    tag_vocab = build_vocab_from_iterator((entry[1] for entry in train_dataset), specials=["<unk>"])

    vocabulary_tks.set_default_index(vocabulary_tks["<unk>"])
    tag_vocab.set_default_index(tag_vocab["<unk>"])

    def collate_batch(batch):
        texts = [b[0] for b in batch]
        tags = [b[1] for b in batch]
        lengths = torch.tensor([len(text) for text in texts])

        texts2 = []
        tags2 = []
        for sentence in texts:
            curr = []
            for word in sentence:
                curr.append(vocabulary_tks[word])
            texts2.append(torch.tensor(curr))

        for tag_list in tags:
            currr = []
            for tag in tag_list:
                currr.append(tag_vocab[tag])
            tags2.append(torch.tensor(currr))

        padded_texts = torch.nn.utils.rnn.pad_sequence(texts2, padding_value=vocabulary_tks["<pad>"], batch_first=True)
        padded_tags = torch.nn.utils.rnn.pad_sequence(tags2, padding_value=tag_vocab["<unk>"], batch_first=True)

        return padded_texts, padded_tags, lengths

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_batch)

    model = BiLSTM(len(vocabulary_tks), embed_dim=300, hidden_dim=256, tagset_size=len(tag_vocab)).to(device)
    train_model(model, train_loader, val_loader, epochs=5)
    
    test_loss, test_accuracy = test(model, test_loader)
    print(f"Testing Loss: {test_loss:.2f}, Testing Accuracy: {test_accuracy:.2f}")

    sentence = "The old man the boat."
    tag_sentence(vocabulary_tks, tag_vocab, model, sentence)
    sentence = "The complex houses married and single soldiers and their families."
    tag_sentence(vocabulary_tks, tag_vocab, model, sentence)
    sentence = "The man who hunts ducks out on weekends."
    tag_sentence(vocabulary_tks, tag_vocab, model, sentence)


if __name__ == "__main__":
    main()