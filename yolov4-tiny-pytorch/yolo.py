import colorsys
import pickle
import os
import time
from typing import List

import numpy as np
import sklearn
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont, Image

from nets.yolo import YoloBody
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image)
from utils.utils_bbox import DecodeBox
import random
import mediapipe as mp
from copy import deepcopy
import cv2
from math import sqrt


def caculate_vector_angle(vector1, vector2):
    '''计算两个向量的夹角，为提升模型的预测效果，使用角度制'''
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    mode_1 = np.sqrt(vector1.dot(vector1))
    mode_2 = np.sqrt(vector2.dot(vector2))
    dot_value = vector1.dot(vector2)

    cos_value = dot_value / (mode_1 * mode_2)
    angle_value = np.arccos(cos_value) # 默认弧度制
    
    # 角度修正
    # 使用角度制，这样数值更接近于 距离特征的值 利于 KNN 模型的预测
    angle_value = angle_value * 180 / np.pi

    return angle_value


def FPS_Mode_FE(item):
    main_keypoint_index = [0, 11, 12, 13, 14, 15, 16, 23, 24]
    # distance_list = list()
    angle_list = list()
    '''
    for i in range(len(main_keypoint_index)):
        for j in range(i+1, len(main_keypoint_index)):
            keypoint_number_1 = main_keypoint_index[i]
            keypoint_number_2 = main_keypoint_index[j]
            # 头到肩膀，头到髋，肩膀到髋的距离是不变的，没必要作为特征
            if keypoint_number_1==0 and keypoint_number_2==12:
                continue
            elif keypoint_number_1==0 and keypoint_number_2==24:
                continue
            elif keypoint_number_1==12 and keypoint_number_2==24:
                continue
            elif keypoint_number_1==0 and keypoint_number_2==11:
                continue
            elif keypoint_number_1==0 and keypoint_number_2==23:
                continue
            elif keypoint_number_1==11 and keypoint_number_2==23:
                continue

            keypoint_1 = item.get(keypoint_number_1)
            keypoint_2 = item.get(keypoint_number_2)

            if keypoint_1 is None or keypoint_2 is None:
                distance = np.nan
            else:
                distance = sqrt((keypoint_1[0] - keypoint_2[0])**2 + (keypoint_1[1] - keypoint_2[1])**2)
            distance_list.append(distance)
    '''
    for i in range(len(main_keypoint_index)):
        for j in range(i+1, len(main_keypoint_index)):
            for k in range(j+1, len(main_keypoint_index)):
                keypoint_number_1 = main_keypoint_index[i]
                keypoint_number_2 = main_keypoint_index[j]
                keypoint_number_3 = main_keypoint_index[k]

                flag = False # 若 flag为True则进行向量夹角的计算
                if keypoint_number_1 == 0 and keypoint_number_2 == 12 and keypoint_number_3 == 14:
                    flag = True
                elif keypoint_number_1 == 12 and keypoint_number_2 == 14 and keypoint_number_3 == 16:
                    flag = True
                elif keypoint_number_1 == 12 and keypoint_number_2 == 14 and keypoint_number_3 == 24:
                    flag = True
                elif keypoint_number_1 == 0 and keypoint_number_2 == 11 and keypoint_number_3 == 13:
                    flag = True
                elif keypoint_number_1 == 11 and keypoint_number_2 == 13 and keypoint_number_3 == 15:
                    flag = True
                elif keypoint_number_1 == 11 and keypoint_number_2 == 13 and keypoint_number_3 == 23:
                    flag = True       

                if flag:
                    keypoint_1 = item.get(keypoint_number_1)
                    keypoint_2 = item.get(keypoint_number_2)
                    keypoint_3 = item.get(keypoint_number_3)

                    if keypoint_1 is None or keypoint_2 is None or keypoint_3 is None:
                        angle_list.append(np.nan)
                    else:
                        vector1 = [keypoint_2[0]-keypoint_1[0], 
                                    keypoint_2[1]-keypoint_1[1], keypoint_2[2]-keypoint_1[2]]
                        vector2 = [keypoint_3[0]-keypoint_1[0], 
                                    keypoint_3[1]-keypoint_1[1], keypoint_3[2]-keypoint_1[2]]

                        angle = caculate_vector_angle(vector1, vector2)
                        angle_list.append(angle*0.01)

    # return [*distance_list, *angle_list]
    return angle_list

class YOLO(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        #--------------------------------------------------------------------------#
        "model_path"        : 'logs/ep100-loss2.389-val_loss3.207.pth',
        "classes_path"      : 'model_data/voc_classes.txt',
        #---------------------------------------------------------------------#
        #   anchors_path代表先验框对应的txt文件，一般不修改。
        #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
        #---------------------------------------------------------------------#
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "anchors_mask"      : [[3,4,5], [1,2,3]],
        #-------------------------------#
        #   所使用的注意力机制的类型
        #   phi = 0为不使用注意力机制
        #   phi = 1为SE
        #   phi = 2为CBAM
        #   phi = 3为ECA
        #-------------------------------#
        "phi"               : 1,  
        #---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        #---------------------------------------------------------------------#
        "input_shape"       : [416, 416],
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"        : 0.1,  # 0.35
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.3,
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : False,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化YOLO
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            
        #---------------------------------------------------#
        #   获得种类和先验框的数量
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.anchors, self.num_anchors      = get_anchors(self.anchors_path)
        self.bbox_util                      = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)

        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

        self.mp_pose = mp.solutions.pose
        # 导入模型
        self.pose = self.mp_pose.Pose(static_image_mode=True,        # 是静态图片还是连续视频帧
                        min_detection_confidence=0.1,
                        min_tracking_confidence=0.5)   # 追踪阈值
        with open("./KNNRegressor.pk", "rb") as fp:
            self.KNNRegressor = pickle.load(fp)
        
        with open("./XGBRegressor.pk", "rb") as ff:
            self.XGBRegressor = pickle.load(ff)

        with open("./fill_nparray.pk", "rb") as f:
            self.fill_nparray = pickle.load(f)

    #---------------------------------------------------#
    #   生成模型
    #---------------------------------------------------#
    def generate(self):
        #---------------------------------------------------#
        #   建立yolo模型，载入yolo模型的权重
        #---------------------------------------------------#
        self.net    = YoloBody(self.anchors_mask, self.num_classes, self.phi)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def detect_image(self, image):
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        # real_img = deepcopy(image)
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return image

            top_label   = np.array(results[0][:, 6], dtype = 'int32') # 0:person 1:arms
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]
        #---------------------------------------------------------#
        #   设置字体与边框厚度
        #---------------------------------------------------------#
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        # 标记哪个人持武器的dict    var:top_label  person:0  arms:1
        person_arms = {i:False for i in range(len(top_label)) if top_label[i]==0}  # dict  key:top_label 人 的下标  value:此人是否持有武器
        # 记录每个人对应的坐标中心点
        person_coo_mean = {i:0 for i in range(len(top_label)) if top_label[i]==0}
        # 标记每个人对应的暴力犯罪指数：连续值 为回归模型预测出来的 pred_value
        person_violence_value = {i:None for i in range(len(top_label)) if top_label[i]==0}

        #---------------------------------------------------------#
        #   图像绘制 之 mediapipe 关键点可视化
        #---------------------------------------------------------#
        for k, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[k]
            score           = top_conf[k]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            
            # label = '{} {:.2f}'.format(predicted_class, score)
            label = '{}'.format(predicted_class)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            # print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])
            

            # for i in range(thickness):
            # 先截取后画框框
            # w, h = image.size
            # region = image.crop([left, h-bottom, right, h-top])  # crop() 指定的是与每个边界的距离
            region = image.crop([left, top, right, bottom])  # crop() 指定的是与每个边界的距离
            
            # from PIL import ImageShow
            # ImageShow.IPythonViewer(region)
            # region = image.crop(box) 

            # image_height, image_width, _ = roi.shape
            
            # TODO fashengshenmeshile
            if predicted_class != "arms":
                # 设置对应的人的坐标中心点为候选框的中心点
                person_coo_mean[k] = (right-left, top-bottom)

                # region.save("./test_blze/balabal{}.jpg".format(g))
                region = cv2.cvtColor(np.asarray(region),cv2.COLOR_RGB2BGR)
                results = self.pose.process(region)

                mp_drawing = mp.solutions.drawing_utils

                keypoint_coordinate = {}  # 关键点列表

                no_keypoint = False  # 若未检测出关键点，则进行预测

                if results.pose_landmarks: # 若检测出人体关键点
                    # 可视化关键点及骨架连线
                    mp_drawing.draw_landmarks(region, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        
                    for i in range(33): # 遍历所有33个关键点，可视化

                        # 获取该关键点的三维坐标
                        x = results.pose_landmarks.landmark[i].x
                        y = results.pose_landmarks.landmark[i].y
                        z = results.pose_landmarks.landmark[i].z
                        # 得到关键点在图像中的像素坐标
                        h, w,_ = region.shape
                        cx = int(x * w)
                        cy = int(y * h)
                        
                        if i not in [*range(27, 33), 
                                    *range(1, 11), 21, 19, 17, 20, 18, 22]:
                            keypoint_coordinate[i] = (cx, cy, z)

                        radius = 5

                        if i == 1: # 眼睛
                            region = cv2.circle(region,(cx,cy), radius, (0,0,255), -1)
                        elif i in [11,12]: # 肩膀
                            region = cv2.circle(region,(cx,cy), radius, (223,155,6), -1)
                        elif i in [23,24]: # 髋关节
                            region = cv2.circle(region,(cx,cy), radius, (1,240,255), -1)
                        elif i in [13,14]: # 胳膊肘
                            region = cv2.circle(region,(cx,cy), radius, (140,47,240), -1)
                        elif i in [25,26]: # 膝盖
                            region = cv2.circle(region,(cx,cy), radius, (0,0,255), -1)
                        elif i in [15,16,27,28]: # 手腕和脚腕
                            region = cv2.circle(region,(cx,cy), radius, (223,155,60), -1)
                        elif i in [17,19,21]: # 左手
                            region = cv2.circle(region,(cx,cy), radius, (94,218,121), -1)
                        elif i in [18,20,22]: # 右手
                            region = cv2.circle(region,(cx,cy), radius, (16,144,247), -1)
                        elif i in [27,29,31]: # 左脚
                            region = cv2.circle(region,(cx,cy), radius, (29,123,243), -1)
                        elif i in [28,30,32]: # 右脚
                            region = cv2.circle(region,(cx,cy), radius, (193,182,255), -1)
                        elif i in [9,10]: # 嘴
                            region = cv2.circle(region,(cx,cy), radius, (205,235,255), -1)
                        # elif i in [1,2,3,4,5,6,7,8]: # 眼及脸颊
                        #     img = cv2.circle(img,(cx,cy), radius, (94,218,121), -1)
                        else: # 其它关键点
                            region = cv2.circle(region,(cx,cy), radius, (0,255,0), -1)

                else:
                    no_keypoint = True
                    print('从图像中未检测出人体关键点，报错。')

                if not no_keypoint:
                    feature_list = FPS_Mode_FE(keypoint_coordinate)
                    # 用平均值填充缺失值
                    for i in range(len(feature_list)):
                        if feature_list[i] is np.nan:
                            feature_list[i] = self.fill_nparray[i]

                    pred_value = self.XGBRegressor.predict(np.array([feature_list]))
                    person_violence_value[k] = pred_value
                    # print(pred_value)
                else:
                    pred_value = None
                    person_violence_value[k] = pred_value

                # TODO 写文档的时候用词截图
                # g = random.randint(1, 100)
                # cv2.imwrite("./mediapipe_test/gg{}.jpg".format(g), region)

                # opencv 对象转 Pillow 的 Image 对象
                region = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
                image.paste(region,[left, top, right, bottom])

        
        # 看武器离哪个人最近，就属于哪个人
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]

            if predicted_class == "arms":
                top, left, bottom, right = box
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))

                arms_coo_mean = (right-left, top-bottom)

                violent_person_key = False
                max_distance = 0

                for key, value in person_coo_mean.items():
                    _distance = (value[0] - arms_coo_mean[0])**2 +\
                                 (value[1] - arms_coo_mean[1])**2
                    if _distance > max_distance:
                        max_distance = _distance
                        violent_person_key = key

                person_arms[violent_person_key] = True  # 标记持有武器的人
        # print(person_arms)


        #---------------------------------------------------------#
        #   图像绘制 之 yolov4 先验框绘制
        #---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            violence_score = None
            if c==0:
                # label = label + str(person_violence_value[i])
                if person_violence_value[i]:
                    violence_score = "{:.2f}".format(person_violence_value[i][0])  # 格式化字符串
                else:
                    violence_score = ""  # 若未空
                label = str(violence_score)

            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')

            # TODO print
            # print(label, top, left, bottom, right)
            


            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # self.colors[c]
            color_person_ord = (0, 255, 0)  # 正常人绘制上绿色
            color_person_warn = (255, 0, 0)  # 危险人员绘制上红色


            warn = None

            if c==0 and person_arms[i]:  # 若此人持有武器
                warn = True

            color_object = None
            if c == 1:
                color_object = self.colors[c]
            elif not warn and c==0:
                color_object = color_person_ord
            elif warn and c==0:
                if person_violence_value[i]:
                    if person_violence_value[i] + 0.5 > 1:
                        color_object = color_person_warn
                    else:
                        color_object = color_person_ord

            # 若判断为超过阈值的人，则框的颜色有绿色转换为红色
            for ii in range(thickness):
                draw.rectangle([left + ii, top + ii, right - ii, bottom - ii], outline=color_object)
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=color_object)
            # label 改为暴力犯罪指数和是否持有刀具棍棒的加权值
            if warn and c==0:
                if person_violence_value[i]:
                    if person_violence_value[i] + 0.5 > 1:
                        label = "warning"
            if label == "warning":
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            else:
                draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
                                                    
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                #---------------------------------------------------------#
                #   将图像输入网络当中进行预测！
                #---------------------------------------------------------#
                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)
                #---------------------------------------------------------#
                #   将预测框进行堆叠，然后进行非极大抑制
                #---------------------------------------------------------#
                results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                            image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
                            
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return 

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
