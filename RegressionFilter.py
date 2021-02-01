import math, cv2
import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from statsmodels import robust
import General as general

FRAME_RATE = general.FRAME_RATE
TIMESTAMP_PATH = general.TIMESTAMP_PATH
REID_PATH = general.REID_PATH
ts_base = general.ts_base
cameras = general.cameras

def get_Regression_Hashmap(res_thres=2):

    reid_hash_map = {cam: {} for cam in cameras}
    exist_max_oid = -1

    ## covernt gt results to camera data hashmap
    #
    # def data_to_hashmap(cam_name_path):
    # 	gt = open(cam_name_path).read().split('\n')[:-1]
    # 	gt_hashmap = {(int(each.split(',')[0]), int(each.split(',')[1])): \
    # 				   tuple([int(i) for i in each.split(',')[2:6]]) for each in gt}
    # 	return gt_hashmap

    # for cam in cameras:
    #     reid_hash_map[cam] = data_to_hashmap("/home/hongpeng/Desktop/research/AICity/Track3/train/S01/" + cam + '/' + 'gt/gt.txt')


    for line in open(REID_PATH).readlines():
        min_cid = 1 if REID_PATH == "./reid_trn.txt" else 6
        cid, oid, frame_id, left, top, width, height = [int(line.split(' ')[i]) for i in range(7)]
        exist_max_oid = max(exist_max_oid, oid)
        reid_hash_map[cameras[cid-min_cid]][(frame_id, oid)] = (left, top, width, height)


    def prepare_regression_data(reid_hash_map, source_cam, destination_cam, time_window):
        source_data, destination_data, source_data_id, destination_data_id = [], [], [], []
        frame_diff = {cam: int(FRAME_RATE * ts_base[cam]) for cam in [source_cam, destination_cam]}
        frame_diff = {cam: frame_diff[cam] - max(frame_diff.values()) for cam in [source_cam, destination_cam]}

        for t, obj in reid_hash_map[source_cam]:
            if t - frame_diff[source_cam] < time_window[0] or t - frame_diff[source_cam] >= time_window[1]:
                continue
            if (t - frame_diff[source_cam] + frame_diff[destination_cam], obj) not in reid_hash_map[destination_cam]:
                continue
            source_data.append(list(reid_hash_map[source_cam][(t, obj)]))
            destination_data.append(list(reid_hash_map[destination_cam][(t - frame_diff[source_cam] + frame_diff[destination_cam], obj)]))
            source_data_id.append((t, obj))
            destination_data_id.append((t - frame_diff[source_cam] + frame_diff[destination_cam], obj))

        return np.array(source_data), np.array(destination_data), source_data_id, destination_data_id


    def frame_obj_to_cameras(reid_hash_map, cameras, time_window):
        fo_to_cams = {}

        frame_diff = {cam: int(FRAME_RATE * ts_base[cam]) for cam in cameras}
        frame_diff = {cam: frame_diff[cam] - max(frame_diff.values()) for cam in cameras}
        for cam in cameras:
            for t, obj in reid_hash_map[cam]:
                if t - frame_diff[cam] < time_window[0] or t - frame_diff[cam] >= time_window[1]:
                    continue
                if (t - frame_diff[cam], obj) not in fo_to_cams:
                    fo_to_cams[(t - frame_diff[cam], obj)] = [cam]
                else:
                    fo_to_cams[(t - frame_diff[cam], obj)].append(cam)

        return fo_to_cams, frame_diff

    fo_to_cams, frame_diff = frame_obj_to_cameras(reid_hash_map, cameras, [0, 900])
    cam_to_outliers = {}

    for source_cam in cameras:
        outlier = set() 
        for destination_cam in cameras:
            if source_cam == destination_cam:
                continue
            source_data, destination_data, source_data_id, destination_data_id = \
                prepare_regression_data(reid_hash_map, source_cam, destination_cam, [0, 900])

            degree = 4 if source_data.shape[0] > 80 else 3
            if source_data.shape[0] > 120: degree = 4
            if source_data.shape[0] > 150: degree = 5
            if source_data.shape[0] > 220: degree = 6

            y_mad = np.linalg.norm(robust.mad(destination_data, axis=0), 2)
            regr = make_pipeline(PolynomialFeatures(degree),linear_model.RANSACRegressor(residual_threshold = res_thres * y_mad))
            regr.fit(source_data, destination_data)
            inlier_mask = regr[-1].inlier_mask_
            outlier_mask = np.logical_not(inlier_mask)

            # y_pred = regr.predict(source_data)
            # print(np.sum(outlier_mask==True), outlier_mask.shape[0], np.sum(outlier_mask==True)/ outlier_mask.shape[0])

            tmp_outlier_pos = list(np.where(outlier_mask == True)[0])
            outlier = outlier.union(set([source_data_id[i] for i in tmp_outlier_pos]))

        cam_to_outliers[source_cam] = outlier

    for cam in cameras:
        for frame, obj in cam_to_outliers[cam]:
            if (frame - frame_diff[cam], obj) in fo_to_cams:
                fo_to_cams[(frame - frame_diff[cam], obj)].remove(cam)

    true_outliers = [key for key in fo_to_cams.keys() if len(fo_to_cams[key]) == 0]

    print('Regression max obj id', exist_max_oid)
    print('Regression Outlier Number', len(true_outliers))

    ## clean reid_hashmap with true_outlier
    for cam in cameras:
        for frame, obj in true_outliers:
            frame = frame + frame_diff[cam]
            if (frame, obj) not in reid_hash_map[cam]:
                continue
            bbox = reid_hash_map[cam][(frame, obj)]
            del reid_hash_map[cam][(frame, obj)]
            exist_max_oid += 1
            reid_hash_map[cam][(frame, exist_max_oid)] = bbox
            

    Multi_Hashmap = reid_hash_map

    return Multi_Hashmap