import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import time, copy
from torch import nn, optim
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image
from torchvision import models


my_trans = transforms.Compose([transforms.Resize(size=[224,224]),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

# code from online
class Dataset(torch.utils.data.Dataset):
    def __init__(self, df=None, transform=None):
        self.df = df
        self.transform = transform
        self.df_len = df.shape[0]
        self.labels = np.unique(df.iloc[:, 1].values).tolist()

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        label = self.df.iloc[index, 1]
        for ind, value in enumerate(self.labels):
            if value == label:
                label = ind

        path_img = self.df.iloc[index, 0]
        img = Image.open(path_img)
        if self.transform:
            img = self.transform(img)

        sample = {'image': img, 'label': label}
        return sample

    def __len__(self):
        return self.df_len

# self
def draw(epochs_training_loss, epochs_validation_loss, epochs_training_accuracy, epochs_val_accuracy):
    plt.figure(figsize=(10, 5), facecolor='w')
    epochs = len(epochs_training_accuracy)
    x_axis_validatoin = np.array(range(epochs)) + 1.
    x_axis_training = np.array(range(epochs)) + 1.
    # loss
    plt.plot(x_axis_validatoin, epochs_validation_loss, 'o-b', label='val loss', lw=2)
    plt.plot(x_axis_training, epochs_training_loss, 'o-y', label='train loss', lw=2)
    # accuracy
    plt.plot(x_axis_validatoin, epochs_val_accuracy, 'o-r', label='val accuracy', lw=2)
    plt.plot(x_axis_training, epochs_training_accuracy, 'o-g', label='train accuracy', lw=2)
    plt.title('loss decreasing and accuracy increasing graph')
    plt.legend()
    plt.show()


path_df_trainset = '/Users/yunhou/course/cs/455/dataset/dataset-master/dataset-master/trainset.csv'
path_df_holdoutset = '/Users/yunhou/course/cs/455/dataset/dataset-master/dataset-master/holdoutset.csv'
df_trainset = pd.read_csv(path_df_trainset)
df_holdoutset = pd.read_csv(path_df_holdoutset)
df_trainset, df_valset = train_test_split(df_trainset, test_size = 0.1)

resnet34 = models.resnet34(pretrained=True)
fc_inputs = resnet34.fc.in_features
resnet34.fc = nn.Sequential(
    nn.Linear(fc_inputs, 10),
    nn.ReLU(),
    nn.Linear(10, 4),
    nn.LogSoftmax(dim=1)
)

# code majority from online
epochs_training_loss = np.array([], dtype='float64')
epochs_validation_loss = np.array([], dtype='float64')


train_set = Dataset(df=df_trainset, transform=my_trans)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=16)
validation_set = Dataset(df=df_valset, transform=my_trans)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=16)
validation_accuracy = []
train_accuracy = []

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet34.parameters())

best_model_value = None
best_epoch_value = 0
loss_best = 9999

for epoch in range(15):
    epoch_start_time = time.time()
    print("Begin the epoch:", epoch)
    train_running_loss = 0
    validation_running_loss = 0

    resnet34.train()

    right_predict = 0
    for batch in train_loader:

        images = batch['image']
        labels = batch['label']

        predictions = resnet34(images)
        loss = loss_function(predictions, labels)

        right_predict += torch.sum(torch.argmax(predictions, axis=1) == labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_running_loss += loss.item()
    train_accuracy.append((right_predict / df_trainset.shape[0]).numpy())

    resnet34.eval()
    right_predict = 0
    for batch in validation_loader:
        images = batch['image']
        labels = batch['label']

        predictions = resnet34(images)  # predict labels
        loss_val = loss_function(predictions, labels)  # calculate the loss
        right_predict += torch.sum(torch.argmax(predictions, axis=1) == labels)
        validation_running_loss += loss_val.item()
    validation_accuracy.append((right_predict / df_valset.shape[0]).cpu().numpy())

    epoch_loss = validation_running_loss / len(validation_loader)
    if loss_best > epoch_loss:
        loss_best = epoch_loss
        best_epoch_value = epoch
        best_model_value = copy.deepcopy(resnet34)

    epochs_training_loss = np.append(epochs_training_loss, train_running_loss / len(train_loader))
    epochs_validation_loss = np.append(epochs_validation_loss, epoch_loss)

    epoch_end = time.time()
    print('\tThe Time consumption is {:.2f}'.format(epoch_end - epoch_start_time))
    print('\tThe Training loss is {:.2f}'.format(train_running_loss / len(train_loader)))
    print('\tThe Validation loss is {:.2f}'.format(epoch_loss))
print('Train just finished, the best epoch is {}.'.format(best_epoch_value))

draw(epochs_training_loss = epochs_training_loss, epochs_validation_loss = epochs_validation_loss,
     epochs_training_accuracy=train_accuracy, epochs_val_accuracy=validation_accuracy)

holdout_set = Dataset(df=df_holdoutset, transform=my_trans)
holdout_loader = torch.utils.data.DataLoader(holdout_set, batch_size=len(holdout_set))

# self
data = next(iter(holdout_loader))
image = data['image']
label = data['label']
prediction = best_model_value(image).detach().numpy()
prediction_class = np.argmax(prediction, axis=1)
label = label.detach().numpy()
num_right_predict = np.sum(prediction_class == label)
num_predict = len(holdout_set)

print('The number of right prediction is {}'.format(num_right_predict))
print('The number of total prediction is {}'.format(num_predict))
print('The accuracy is {:.3f}'.format(num_right_predict / num_predict))


