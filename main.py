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
    composed_trn = transforms.Compose([ResizeShorterScale(shorter_side, low_scale, high_scale),
                                       Pad(crop_size, [123.675, 116.28, 103.53], ignore_label),
                                       RandomMirror(),
                                       RandomCrop(crop_size),
                                       Normalise(*normalise_params),
                                       ToTensor()])
    composed_val = transforms.Compose([Normalise(*normalise_params),
                                       ToTensor()])
    return composed_trn,composed_val

def train_prapre():
    trainset = MyDataset(data_file=train_list,
                         data_dir=train_dir,
                         transform_trn=composed_trn,
                         transform_val=composed_val)
    valset = MyDataset(data_file=val_list,
                       data_dir=val_dir,
                       transform_trn=None,
                       transform_val=composed_val)
    return trainset,valset
# 导入数据集

def data_loader():
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
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
