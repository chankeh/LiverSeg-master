import cv2 as cv
import numpy as np
import PIL.Image as Image
import os

np.random.seed(42)


def split_dataset():
    # 读取图像文件
    images_path = "./liver/train/data_train/"
    images_list = os.listdir(images_path)  # 每次返回文件列表顺序不一致
    images_list.sort()  # 需要排序处理
    images_list = [images_path+x for x in images_list]

    # 读取标签/Mask图像
    labels_path = "./liver/train/label/"
    labels_list = os.listdir(labels_path)
    labels_list.sort()
    labels_list = [labels_path+x for x in labels_list]

    # 读取验证集文件
    val_images_path = './liver/val/data_val'
    val_images_list = os.listdir(val_images_path)
    val_images_list.sort()
    val_images_list = [val_images_path + x for x in val_images_list]

    # 读取验证集标签mask
    val_label_path = './liver/val/label'
    val_label_list = os.listdir(val_label_path)
    val_label_list.sort()
    val_label_list = [val_label_path + x for x in val_label_list]

    # 创建路径文件 (使用二进制编码, 避免操作系统不匹配)
    train_file = "./train.data"
    test_file = "./test.data"
    # if os.path.isfile(train_file) and os.path.isfile(test_file):
    #     return
    # train_file = open(train_file, "wb")
    # test_file = open(test_file, "wb")
    #
    # # 划分数据集
    # split_ratio = 0.8
    # for image, label in zip(images_list, labels_list):
    #     image = os.path.join(images_path, image)
    #     label = os.path.join(labels_path, label)
    #     if os.path.basename(image).split('.')[0] != os.path.basename(label).split('.')[0]:
    #         continue
    #     file = train_file if np.random.rand() < split_ratio else test_file
    #     file.write((image + "\t" + label + "\n").encode("utf-8"))
    # train_file.close()
    # test_file.close()
    print("成功读取数据集!")
    return images_list,labels_list,val_images_list,val_label_list


def read_image(path):
    img = np.array(Image.open(path))
    if img.ndim == 2:
        img = cv.merge([img, img, img])
    return img


def test_read(images_list,labels_list):
    train_file = "./liver/train/data_train"
    # with open(train_file, 'rb') as f:
    #     datalist = f.readlines()
    # datalist = [(k, v) for k, v in map(lambda x: x.decode('utf-8').strip('\n').split('\t'), zip(images_list,labels_list))]
    datalist = [[k,v] for k,v in zip(images_list,labels_list)]
    item = datalist[0]
    image = read_image(item[0])
    mask = read_image(item[1])
    cv.imshow("image", image)
    cv.imshow("mask", mask)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    images_list,labels_list,val_images_list,val_label_list = split_dataset()
    test_read(images_list,labels_list)
