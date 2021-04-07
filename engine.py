import torch
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from src import config
from src import dataset as d
from src import model
import torch.nn as nn
from tqdm import tqdm
df = pd.read_csv(config.file_path)
import numpy as np
torch.cuda.empty_cache()

reviews,labels = df["review"],df["sentiment"]
rev = []
lab= []
for i,j in d.Dataet(reviews,labels):
    rev.append(i)
    lab.append(j)

X_train,X_val,y_train,y_val = train_test_split(rev,lab,test_size=0.1,random_state=24,stratify=labels)

X_tr,y_tr,X_te,y_te,oneHotdict = config.tokenize(X_train,y_train,X_val,y_val)
X_tr = config.padding(X_tr,config.len_sez)
X_te = config.padding(X_te,config.len_sez)
train_X = X_tr[:int(0.90*len(X_tr))]
train_Y = y_tr[:int(0.90*len(X_tr))]
val_X = X_tr[int(0.90*len(X_tr)):]
val_y = y_tr[int(0.90*len(X_tr)):]

train_data = TensorDataset(torch.from_numpy(train_X),torch.from_numpy(train_Y))
val_data = TensorDataset(torch.from_numpy(val_X),torch.from_numpy(val_y))

test_data = TensorDataset(torch.from_numpy(X_te),torch.from_numpy(y_te))


Train_loader = DataLoader(train_data,batch_size=config.batch_size,shuffle=True)
Valid_loader = DataLoader(val_data,batch_size=config.batch_size,shuffle=True)
Test_loader = DataLoader(test_data,batch_size=config.batch_size,shuffle=True)






embedding_dim = 500
hidden_dim = 256
n_layers = 2



counter = 0
grad_clip = 5
printing = 100


# Instantiate the model w/ hyperparams
vocab_sz = len(oneHotdict) + 1
output_size = 17

n_layers = 2
train_on_gpu = torch.cuda.is_available()
model = model.RNNSentiment(num_layer=config.n_layers,vocab_size=vocab_sz,hidden_dim=config.hidden_dim,embedding_dim=config.embedding_dim)
# loss and optimization functions
lr=0.001

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# training params
epochs = 10
counter = 0
print_every = 1000
clip=5 # gradient clipping

# move model to GPU, if available
if(train_on_gpu):
    model.cuda()

for e in range(epochs):
    model.train()
    # batch loop
    for inputs, labels in tqdm(Train_loader,total=len(Train_loader)):
        counter += 1
        batch_size = inputs.shape[0]
        h = model.init_hidden(batch_size)
        if(train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        model.zero_grad()

        # get the output from the model
        output, h = model(inputs, h)
        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if counter % print_every == 0:
            # Get validation loss

            val_losses = []
            model.eval()

            for inputs, labels in Valid_loader:
                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                batch_siz = inputs.shape[0]
                val_h = model.init_hidden(batch_siz)

                val_h = tuple([each.data for each in val_h])

                if(train_on_gpu):
                    inputs, labels = inputs.cuda(), labels.cuda()

                output, val_h = model(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())

                val_losses.append(val_loss.item())

                model.train()
print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))




test_loss_con = []
num_correct = 0


for inp,lab in tqdm(Test_loader,total=len(Test_loader)):
    model.cuda()
    inp.cuda()
    lab.cuda()
    model.eval()
    batch_siz = inp.shape[0]
    ht = model.init_hidden(batch_siz)
    ht = tuple([each.data for each in ht])

    output , ht = model(inp,ht)

    test_loss = criterion(output.squeeze(),lab.float())

    test_loss_con.append(test_loss)

    pred = torch.round(output.squeeze())

    correct_tensor = pred.eq(lab.float().view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)


print("Test loss: {:.3f}".format(np.mean(test_loss_con)))

# accuracy over all test data
test_acc = num_correct/len(Test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))
