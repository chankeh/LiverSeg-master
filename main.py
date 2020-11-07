from torch.utils.data import DataLoader
from torchvision import transforms
from data_pack import MyDataset
from data_process import split_dataset

# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

images_list,labels_list,val_images_list,val_label_list = split_dataset()


def compose_data():
    # 定义Transform
    composed_trn = transforms.Compose([transforms.Resize(128),transforms.ToTensor(),transforms.Normalize(0.5,0.5,0.5)])
    composed_val = transforms.Compose([transforms.Normalise(0.5,0.5,0.5),transforms.ToTensor()])
    return composed_trn,composed_val

def train_prapre(composed_trn,composed_val):
    trainset = MyDataset(data_file='./liver/train/data_train/',
                         data_dir='./liver/train/',
                         transform_trn=composed_trn,
                         transform_val=composed_val)
    valset = MyDataset(data_file='./liver/val/data_val',
                       data_dir='./liver/val/',
                       transform_trn=None,
                       transform_val=composed_val)
    return trainset,valset
# 导入数据集

def data_loader(trainset,valset,batch_size,num_workers):
    # 构建生成器
    train_loader = DataLoader(trainset,
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
    return train_loader,val_loader


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
