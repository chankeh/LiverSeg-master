from torch.utils.data import DataLoader
from torchvision import transforms
from data_pack import MyDataset
from data_process import split_dataset
from net_framework import AttentionUNet2D
import torch


# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

images_list,image_labels_list,val_images_list,val_label_list = split_dataset()


def compose_data():
    # 定义Transform
    composed_trn = transforms.Compose([transforms.Resize(128),transforms.ToTensor(),transforms.Normalize(0.5,0.5,0.5)])
    composed_val = transforms.Compose([transforms.Normalize(0.5,0.5,0.5),transforms.ToTensor()])
    return composed_trn,composed_val

#                 composed_trn, composed_val,images_list,labels_list,val_images_list,val_label_list
def train_prapre(composed_trn,composed_val,images_list,imamge_labels_list,val_images_list,val_label_list):
    trainset = MyDataset(data_file=images_list,
                         data_dir=image_labels_list,
                         transform_trn=composed_trn,
                         transform_val=composed_val)
    valset = MyDataset(data_file=val_images_list,
                       data_dir=val_label_list,
                       transform_trn=None,
                       transform_val=composed_val)

    train_labelset = MyDataset(data_file=imamge_labels_list,
                         data_dir='./liver/train/label/',
                         transform_trn=composed_trn,
                         transform_val=composed_val)
    val_labelset = MyDataset(data_file=val_label_list,
                       data_dir='./liver/val/label',
                       transform_trn=None,
                       transform_val=composed_val)

    return trainset,valset,train_labelset,val_labelset
# 导入数据集

def data_loader(trainset,valset,train_labelset,val_labelset,batch_size,num_workers):
    # 构建生成器
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)
    train_label_loader = DataLoader(train_labelset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(valset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True)
    val_label_loader = DataLoader(val_labelset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True)

    return train_loader,val_loader,train_label_loader,val_label_loader

def train_net(model,
              train_loader,
              val_loader,
              train_label_loader,
              val_label_loader,
              n_epochs,
              batch_size,
              weight,
              checkpoint_dir='./weights',
              lr=1e-4):
    pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 1.划分数据集
    images_list,labels_list,val_images_list,val_label_list=split_dataset()
    # 2.定义Transform
    composed_trn, composed_val = compose_data()
    # 3.准备训练所需数据
    trainset,valset,train_labelset,val_labelset = train_prapre(composed_trn, composed_val,images_list,labels_list,val_images_list,val_label_list)
    # 4.构建生成器
    train_loader,val_loader,train_label_loader,val_label_loader = data_loader(trainset,valset,train_labelset,val_labelset,4,2)
    # 5.train
    model = AttentionUNet2D(n_channels=1, n_classes=3)
    for i in range(20):
        train_loss = 0.
        train_acc = 0.
        for i, sample in enumerate(train_loader):
            image = sample['image']
            target = sample['mask']
            image_var = torch.autograd.Variable(image).float()
            target_var = torch.autograd.Variable(target).long()

            optimizer = torch.optim.Adam(model.parameters())
            loss_func = torch.nn.CrossEntropyLoss()

            out = model(image_var)
            loss = loss_func(out, target_var)

            train_loss += loss.data[0]
            pred = torch.max(out, 1)[1]
            train_correct = (pred == target_var).sum()
            train_acc += train_correct.data[0]


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
                images_list)), train_acc / (len(labels_list))))










# See PyCharm help at https://www.jetbrains.com/help/pycharm/
