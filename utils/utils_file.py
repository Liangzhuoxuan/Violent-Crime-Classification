import os
import shutil


def rename_files(new_path= 'F:/毕业设计图片数据收集/JPEGImages/'
            ,path= 'F:/毕业设计图片数据收集/test_rename_file/', copy=False, start=0):
    '''
    批量重命名文件，指定需批量重命名文件目录所在路径
    copy 参数为 True 表示是否需要把当前一批文件更名后复制到另一个目录里面
    start 参数用于进行增量图片处理，指定其图片文件名的其实值
    '''
    for root, dirs, files in os.walk(path):
        for index, filename in enumerate(files, start=start):
            new_filename = None
            if index + 1 < 10:
                new_filename = "00" + str(index+1) + ".jpg"
            elif index + 1 < 100:
                new_filename = "0" + str(index+1) + ".jpg"
            elif index + 1 < 1000:
                new_filename = str(index+1) + ".jpg"
            os.rename(path + filename, path + new_filename)
    if copy:
        shutil.copy(path + new_filename, new_path + new_filename)


if __name__ == "__main__":
    path = "C:/Users/Liang/Desktop/增量图片4/"
    # rename_files(path=path)
    # path = "C:/Users/Liang/Desktop/增量图片3/"
    new_path = "F:/毕业设计/yolov4-tiny-pytorch-master/VOCdevkit/VOC2007/JPEGImages/"
    rename_files(new_path, path, start=80) # 232