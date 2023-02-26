import numpy as np
from sklearn.metrics import mean_absolute_error
from BlazePose import *
from math import sqrt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb


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


def FeatureEng(mode = 0, img=None):
    '''
    @args: mode=0 表示根据本地图片的数据进行特征工程
           mode=1 表示传入一张新的图片，对其进行特征工程

    利用 关键点坐标进行特征创建
    1. 根据关键点坐标之间的欧式距离
    2. 计算主干之间的夹角
    3. 对两者的值进行归一化

    返回 二维矩阵为特征矩阵, 标签列表
    '''
    # if mode == 1:
    #     pass


    with open("./dump_results_dict_process.pk", "rb") as fp:
        dump_results = pickle.load(fp)

    # label_list = [0.5, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.7, 0.7, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 
    #                 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.8, 0.2, 0.2, 0.2, 0.2,
    #                  0.2, 0.2, 0.2, 0.2,0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6]

    label_list = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 
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


    feature_array = list()
    # main_keypoint_index = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26]
    main_keypoint_index = [0, 11, 12, 13, 14, 15, 16, 23, 24]
    # main_keypoint_index = [0, 12, 14, 16, 24] # 11, 13, 15, 23 只使用右手部分的特征
    '''
    for item in dump_results:
        # x = [item[i][0] for i in range(len(item))] 
        # print(x)
        distance_list = list()  # 记录两个点之间欧式距离的向量
        angle_list = list()  # 记录量向量之间夹角 的向量

        # 计算两两坐标之间的距离
        # 计算指定三个点之间的两个向量所成的夹角
        for i in range(len(item)):  # 每个关键点坐标   item: tuple(keypoint_index, x, y, z)
            for j in range(i+1, len(item)):
                keypoint_1 = item[i]
                keypoint_2 = item[j]

                distance = sqrt( (keypoint_1[1] - keypoint_2[1])**2 + (keypoint_1[2] - keypoint_2[2])**2 )
                distance_list.append(distance)
        # print(distance_list.__len__())

        for i in range(len(item)):
            # if i not in [16, 14, 12, 11, 13, 15, 24, 23, 26, 25]:
            #     continue
            for j in range(i+1, len(item)):
                keypoint_1 = item[i]
                keypoint_2 = item[j]
                # x, y, z
                vector1 = [keypoint_1[1], keypoint_1[2], keypoint_1[3]]
                vector2 = [keypoint_2[1], keypoint_2[2], keypoint_2[3]]
                _angle = caculate_vector_angle(vector1, vector2)
                angle_list.append(_angle)

        feature_list = [*distance_list, *angle_list]
        feature_array.append(feature_list)
    '''
    # print(feature_array[1]) # 每个特征数值距离不会太大，适合使用KNN模型进行预测

    for item in dump_results:  # item: dict_object
        angle_list = list()
        '''
        distance_list = list()
        

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

        # feature_array.append([*distance_list, *angle_list])
        feature_array.append(angle_list)

    return feature_array, label_list

def FPS_Mode_FE(item):
    main_keypoint_index = [0, 11, 12, 13, 14, 15, 16, 23, 24]
    distance_list = list()
    angle_list = list()

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

    return [*distance_list, *angle_list]

def train_regression_model():
    '''训练 KNN regression model'''
    feature_array, label_list = FeatureEng()
    with open("./malaoshi.pk", "wb") as fp:
        pickle.dump(feature_array, fp)
    # X_train, X_test, y_train, y_test = train_test_split(feature_array, label_list, random_state=2022)
    # neigh = KNeighborsRegressor(n_neighbors=5)
    # neigh.fit(X_train, y_train)

    # pred_value = list()
    # pred = neigh.predict(X_test)
    # print(pred)
    '''
    param_grid = [{"n_neighbors":[1, 2, 3, 4, 5]}]
    neigh = KNeighborsRegressor(n_neighbors=5)
    grid_search = GridSearchCV(neigh, param_grid, cv=10,
                          scoring='neg_mean_absolute_error')

    grid_search.fit(feature_array, label_list)
    best_es = grid_search.best_estimator_
    print(best_es)
    '''
    feature_array = np.array(feature_array)

    model = xgb.XGBRegressor()
    model.fit(feature_array, label_list)

    # 保存模型
    # with open("KNNRegressor.pk", "wb") as fp:
    #     pickle.dump(best_es, fp)

    with open("XGBRegressor.pk", "wb") as fp:
        pickle.dump(model, fp)


def predict_regression_value():
    '''使用 KNN regression 预测其犯罪倾向指数'''
    pass

if __name__ == "__main__":
    # FeatureEng()
    train_regression_model()