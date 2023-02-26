import os
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import pickle

# from utils.utils_file import rename_files


# 定义可视化图像函数
def look_img(img):
    '''opencv读入图像格式为BGR，matplotlib可视化格式为RGB，因此需将BGR转RGB'''
    # img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()

def blazepose(img_path) -> dict:
    '''
    使用 BlazePose 算法计算一张图片对应的关键点坐标，
    返回值格式为 [(关键点序号, 关键点x坐标, 关键点y坐标, 关键点z坐标),]
    '''
    mp_pose = mp.solutions.pose
    # # 导入绘图函数
    mp_drawing = mp.solutions.drawing_utils 

    # 导入模型
    pose = mp_pose.Pose(static_image_mode=True,        # 是静态图片还是连续视频帧
                        min_tracking_confidence=0.5)   # 追踪阈值

    # 从图片文件读入图像，opencv读入为BGR格式，用cv2.imread() 对于.jpg 格式的图片会返回 NoneType
    # img = cv2.imread('IMG_20220129_105417.jfif')
    img = plt.imread(img_path)
    # 计算图片的 长宽
    h = img.shape[0]
    w = img.shape[1]

    # BGR转RGB
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 将RGB图像输入模型，获取预测结果
    results = pose.process(img_RGB)
    # keypoint_coordinate = list()
    keypoint_coordinate = {}  # key:keypoint_index  value:(x, y, z)

    if results.pose_landmarks: # 若检测出人体关键点
        # 可视化关键点及骨架连线
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        for i in range(33): # 遍历所有33个关键点，可视化

            # 获取该关键点的三维坐标
            x = results.pose_landmarks.landmark[i].x
            y = results.pose_landmarks.landmark[i].y
            z = results.pose_landmarks.landmark[i].z
            # 得到关键点在图像中的像素坐标
            cx = int(x * w)
            cy = int(y * h)
            '''关键点用于构建特征，只需其中一部分
            除去手指、鼻子、眼睛、嘴巴、踝关节，留下主体躯干
            留下 11个关键点，后面计算一张图片的各关键点欧式距离的计算规模就是 11x11=121
            '''
            if i not in [*range(27, 33), 
                        *range(1, 11), 21, 19, 17, 20, 18, 22, 25, 26]:
                # keypoint_coordinate.append((i, x, y, z))
                keypoint_coordinate[i] = (cx, cy, z)

            radius = 10

            if i == 1: # 眼睛
                img = cv2.circle(img,(cx,cy), radius, (0,0,255), -1)
            elif i in [11,12]: # 肩膀
                img = cv2.circle(img,(cx,cy), radius, (223,155,6), -1)
            elif i in [23,24]: # 髋关节
                img = cv2.circle(img,(cx,cy), radius, (1,240,255), -1)
            elif i in [13,14]: # 胳膊肘
                img = cv2.circle(img,(cx,cy), radius, (140,47,240), -1)
            elif i in [25,26]: # 膝盖
                img = cv2.circle(img,(cx,cy), radius, (0,0,255), -1)
            elif i in [15,16,27,28]: # 手腕和脚腕
                img = cv2.circle(img,(cx,cy), radius, (223,155,60), -1)
            elif i in [17,19,21]: # 左手
                img = cv2.circle(img,(cx,cy), radius, (94,218,121), -1)
            elif i in [18,20,22]: # 右手
                img = cv2.circle(img,(cx,cy), radius, (16,144,247), -1)
            elif i in [27,29,31]: # 左脚
                img = cv2.circle(img,(cx,cy), radius, (29,123,243), -1)
            elif i in [28,30,32]: # 右脚
                img = cv2.circle(img,(cx,cy), radius, (193,182,255), -1)
            elif i in [9,10]: # 嘴
                img = cv2.circle(img,(cx,cy), radius, (205,235,255), -1)
            # elif i in [1,2,3,4,5,6,7,8]: # 眼及脸颊
            #     img = cv2.circle(img,(cx,cy), radius, (94,218,121), -1)
            else: # 其它关键点
                img = cv2.circle(img,(cx,cy), radius, (0,255,0), -1)

        # 展示图片
        # look_img(img)
        
    else:
        print('从图像中未检测出人体关键点，报错。')

    # 保存图片
    # cv2.imwrite('D.jpg',img)

    return keypoint_coordinate


def get_each_photo_keypoint_coo():
    '''将数据集里每张图片对应关键点特征 序列化 保存到本地'''
    # binary_cls_data_path = "F:/毕业设计/BinaryClsData/"
    binary_cls_data_path = "F:/毕业设计/RessionImg/"

    # 每张图片对一个表示 犯罪倾向程度 的浮点数
    # trend_degree = [0.5, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.7, 0.7, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 
    #                 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.8, 0.2, 0.2, 0.2, 0.2,
    #                  0.2, 0.2, 0.2, 0.2,0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6]

    trend_degree = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 
                    0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 
                    0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 
                    0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 
                    0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 
                    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 
                    0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 
                    0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 
                    0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 
                    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                    0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 
                    0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 
                    0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 
                    0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 
                    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 
                    0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,]

    dump_results = list()
    
    for root, dirs, files in os.walk(binary_cls_data_path):
        for index, filename in enumerate(files):
            path = binary_cls_data_path + filename
            current_photo_keypoint_coo = blazepose(path)
            # 在 每张图片对应的 list 数据结构的列表首部 插入表示此图片对应的 犯罪倾向指数
            # current_photo_keypoint_coo.insert(0, trend_degree[index])
            # print(current_photo_keypoint_coo)
            dump_results.append(current_photo_keypoint_coo)
    
    with open("dump_results_dict.pk", "wb") as fp:
        pickle.dump(dump_results, fp)

def find_photo_keypoint():
    pass

if __name__ == "__main__":
    get_each_photo_keypoint_coo()