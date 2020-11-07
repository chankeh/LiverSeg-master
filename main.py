from torch.utils.data import DataLoader
from torchvision import transforms
from data_pack import MyDataset
from data_process import split_dataset
from net_framework import AttentionUNet2D
import torch


# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

images_list,imamge_labels_list,val_images_list,val_label_list = split_dataset()


def compose_data():
    # 定义Transform
    composed_trn = transforms.Compose([transforms.Resize(128),transforms.ToTensor(),transforms.Normalize(0.5,0.5,0.5)])
    composed_val = transforms.Compose([transforms.Normalise(0.5,0.5,0.5),transforms.ToTensor()])
    return composed_trn,composed_val

def train_prapre(composed_trn,composed_val,images_list,imamge_labels_list,val_images_list,val_label_list):
    trainset = MyDataset(data_file=images_list,
                         data_dir='./liver/train/data_train/',
                         transform_trn=composed_trn,
                         transform_val=composed_val)
    valset = MyDataset(data_file=val_images_list,
                       data_dir='./liver/val/data_val/',
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








    # Model on cuda
    # if torch.cuda.is_available():
    #     model = model.cuda()

    # data_transform = transforms.RandomHorizontalFlip()

    train_dataset = NrrdReader3D(train_data_path, train_label_path)
    val_dataset = NrrdReader3D(val_data_path, val_label_path)

    label_dataset = NrrdReader3D(train_label_path)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    label_dataloader = DataLoader(train_label_path,batch_size=batch_size, shuffle=True)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
    '''.format(n_epochs, batch_size, lr, train_dataset.__len__(),
               val_dataset.__len__()))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    # criterion = nn.NLLLoss(weight=weight)
    criterion = SoftDiceLoss(n_classes=3)

    losses = []
    val_losses = []
    for epoch in range(n_epochs):
        losses_avg = train_epoch(model,
                                  label_dataloader,
                                 optimizer,
                                 criterion,
                                 epoch,
                                 n_epochs,
                                 print_freq=100)

        val_losses_avg = val_epoch(model,
                                   val_dataloader,
                                   criterion,
                                   print_freq=10)

        losses.append(round(losses_avg.cpu().numpy().tolist(), 4))
        val_losses.append(round(val_losses_avg.cpu().numpy().tolist(), 4))

        # save model parameters
        parameters_name = str(epoch) + '.pkl'
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, parameters_name))

    # save loss figure
    draw_loss(n_epochs, losses, val_losses)

    # save loss data
    with open('loss/loss.txt', 'w') as loss_file:
        loss_file.write('train loss:\n')
        for i, loss in enumerate(losses):
            output = '{' + str(i) + '}: {' + str(loss) + '}\n'
            loss_file.write(output)
        loss_file.write('-' * 50)
        loss_file.write('\n')
        loss_file.write('validation loss:\n')
        for i, val_loss in enumerate(val_losses):
            output = '{' + str(i) + '}: {' + str(val_loss) + '}\n'
            loss_file.write(output)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 1.划分数据集
    images_list,labels_list,val_images_list,val_label_list=split_dataset()
    # 2.定义Transform
    composed_trn, composed_val = compose_data()
    # 3.准备训练所需数据
    trainset,valset = train_prapre(composed_trn, composed_val)
    # 4.构建生成器
    train_loader,val_loader = data_loader(trainset,valset,4,2)
    # 5.train
    for i, sample in enumerate(train_loader):
        image = sample['image'].cuda()
        target = sample['mask'].cuda()
        image_var = torch.autograd.Variable(image).float()
        target_var = torch.autograd.Variable(target).long()
        # Compute output
        net = AttentionUNet2D(n_channels=1, n_classes=2)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
