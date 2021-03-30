import math, cv2
from time import time
import numpy as np
from sklearn import svm
from matplotlib import pyplot as plt
import General as general
import RegressionFilter

# environment & macro definations
FRAME_RATE = general.FRAME_RATE
WORKSAPCE = general.WORKSAPCE
DATA_PATH = general.DATA_PATH
DET_PATH =  general.DET_PATH
ts_base = general.ts_base
cameras = general.cameras
cameras_shape = general.cameras_shape


def prepare_svm_data(reid_hash_map, source_cam, destination_cam, time_window):
    svm_data, svm_label, data_id = [], [], []
    frame_diff = {cam: int(FRAME_RATE * ts_base[cam]) for cam in [source_cam, destination_cam]}
    frame_diff = {cam: frame_diff[cam] - max(frame_diff.values()) for cam in [source_cam, destination_cam]}

    for t, obj in reid_hash_map[source_cam]:
        if t - frame_diff[source_cam] < time_window[0] or t - frame_diff[source_cam] >= time_window[1]:
            continue
        svm_data.append(reid_hash_map[source_cam][(t, obj)])
        data_id.append((t, obj))
        label_data = 0
        for off_set in range(1):
            if (t - frame_diff[source_cam] + frame_diff[destination_cam] + off_set, obj) in reid_hash_map[destination_cam]:
                label_data = 1
        svm_label.append(label_data)

    print(source_cam, destination_cam, sum(svm_label))
    return np.array(svm_data), np.array(svm_label), data_id


def get_SVM_HashMap(gamma=2e-3, res_thres=2.0):

    reid_hash_map = RegressionFilter.get_Regression_Hashmap(res_thres=res_thres)
    exist_max_oid = -1

    for cam_name in reid_hash_map:
        for _, obj in reid_hash_map[cam_name]:
            exist_max_oid = max(exist_max_oid, obj)
    
    # Logic to add some extra bbox from yolo detection

    for cam_name in cameras:
        time_to_bbox = {}
        for frame_id, oid in reid_hash_map[cam_name]:
            if frame_id not in time_to_bbox:
                time_to_bbox[frame_id] = [reid_hash_map[cam_name][(frame_id, oid)]]
            else:
                time_to_bbox[frame_id].append(reid_hash_map[cam_name][(frame_id, oid)])
        
        for line in open(WORKSAPCE + DATA_PATH + cam_name + '/' + DET_PATH).readlines():
            frame_id, _, left, top, width, height, confidence, _, _, _ = [float(each) for each in line.split(',')]
            if confidence < 0.05: continue
            if left < 50 or top < 50 or left + width + 50 > cameras_shape[cam_name][1] or \
                top + height + 50 > cameras_shape[cam_name][0]: continue
            frame_id, left, top, width, height = round(frame_id), round(left), round(top), round(width), round(height)

            add_this = True
            if frame_id in time_to_bbox:
                for exist_bbox in time_to_bbox[frame_id]:
                    if general.IoU(exist_bbox, (left, top, width, height)) > 0.1:
                        add_this = False
                        break
            if add_this:
                exist_max_oid += 1
                reid_hash_map[cam_name][(frame_id, exist_max_oid)] = (left, top, width, height)


    outlier_dict = {cam: [] for cam in cameras}
    unique_dict = {cam: [] for cam in cameras}

    print("SVM function gamma input ", gamma)

    for source_cam in cameras:
        source_unique = set(reid_hash_map[source_cam].keys())
        source_outlier = set()
        for destination_cam in cameras:
            if source_cam == destination_cam:
                continue
            svm_data, svm_label, data_id = prepare_svm_data(reid_hash_map, source_cam, destination_cam, [0, 900])       
            false_pos = list(np.where(svm_label == 0)[0])
            source_unique = source_unique.intersection(set([data_id[i] for i in false_pos]))

            clf = svm.SVC(kernel='rbf', class_weight='balanced', gamma=gamma)  # change to 2e-5 in overall setup
            clf.fit(svm_data, svm_label)
            y_pred = clf.predict(svm_data)

            false_negative_pos = list(np.where((svm_label - y_pred) == -1)[0])
            if len(false_negative_pos) == 0: continue
            tmp_outlier = set([data_id[i] for i in false_negative_pos])
            source_outlier = source_outlier.union(tmp_outlier)    
        outlier_dict[source_cam] = source_unique.intersection(source_outlier)
        unique_dict[source_cam] = source_unique

    outlier_num = 0

    Multi_Hashmap = {}
    for cam in cameras:
        for key in outlier_dict[cam]:
            del reid_hash_map[cam][key]
            outlier_num += 1
        Multi_Hashmap[cam] = reid_hash_map[cam]

    print('SVM Outliers Number', outlier_num)

    return Multi_Hashmap, reid_hash_map,  outlier_dict, unique_dict


if __name__ == "__main__":

    Multi_Hashmap, reid_hash_map, outlier_dict, unique_dict = get_SVM_HashMap(gamma=100, res_thres=100)

    for cam in outlier_dict:
        print(len(outlier_dict[cam]))

    image_frame_id = {'c001': 630, 'c002': 630, 'c003': 630, 'c004': 630, 'c005': 630}
    # image_frame_id = {'c006': 630, 'c007': 630, 'c008': 630, 'c009': 630}

    unique_area = []
    total_area = []

    for cam in cameras:
        cam_shape =  (general.cameras_shape[cam][1], general.cameras_shape[cam][0], 3)
        base_frame = np.zeros(cam_shape).astype(np.uint8)
        dark_frame = np.zeros(cam_shape).astype(np.uint8)
        for key in unique_dict[cam] - outlier_dict[cam] :
            left, top, width, height = reid_hash_map[cam][key]
            base_frame[left: left+width, top: top+height] = 255

        unique_area.append(np.sum(base_frame) / (255 * 3))
        total_area.append(cam_shape[0] * cam_shape[1])

        base_frame = np.moveaxis(base_frame, 0, 1)
        dark_frame = np.moveaxis(dark_frame, 0, 1)

        image_frame = general.get_frame(cam, image_frame_id[cam])
        print(image_frame.shape, dark_frame.shape)
        combine_frame = cv2.addWeighted( dark_frame, 0.8, image_frame, 0.5, 1)

        transparent_loc = tuple(np.where(base_frame == 255))
        combine_frame[transparent_loc] = image_frame[transparent_loc]

        print(unique_area[-1] / total_area[-1])

        cv2.imshow('image', combine_frame)
        cv2.waitKey()
        cv2.destroyAllWindows()

    print(sum(unique_area) / sum(total_area))