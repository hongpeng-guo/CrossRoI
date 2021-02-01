import math, cv2

from numpy.lib.histograms import _search_sorted_inclusive
import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from statsmodels import robust
import General as general
import Evaluate as eva

FRAME_RATE = general.FRAME_RATE
WORKSAPCE = general.WORKSAPCE
DATA_PATH = general.DATA_PATH
DET_PATH =  general.DET_PATH
TIMESTAMP_PATH = general.TIMESTAMP_PATH
REID_PATH = general.REID_PATH
ts_base = general.ts_base
cameras = general.cameras
frame_diff = general.frame_diff
cameras_shape = general.cameras_shape

# covernt gt results to camera data hashmap


gt_hash_map = {cam: {} for cam in cameras}
gt_time_obj = {cam: {} for cam in cameras}

def data_to_hashmap(cam_name_path):
	gt = open(cam_name_path).read().split('\n')[:-1]
	gt_hashmap = {(int(each.split(',')[0]), int(each.split(',')[1])): \
				   tuple([int(i) for i in each.split(',')[2:6]]) for each in gt}
	return gt_hashmap

for cam in cameras:
    gt_time_to_obj_id = eva.time_to_gt_bbox_obj(cam)
    gt_time_to_obj_id = eva.gt_time_bbox_obj_Kalman_filter(cam, gt_time_to_obj_id)
    for time in gt_time_to_obj_id:
        for rect_obj in gt_time_to_obj_id[time]:
            rect, obj = rect_obj[:4], rect_obj[-1]
            gt_hash_map[cam][(time, obj)] = rect


for cam in cameras:
    for time, id in gt_hash_map[cam].keys():
        if time not in gt_time_obj[cam]:
            gt_time_obj[cam][time] = set([id])
        else:
            gt_time_obj[cam][time].add(id)

reid_hash_map = {cam: {} for cam in cameras}
reid_time_obj = {cam: {} for cam in cameras}

exist_max_oid = -1
for line in open(REID_PATH).readlines():
    min_cid = 1 if REID_PATH == "./reid_trn.txt" else 6
    cid, oid, frame_id, left, top, width, height = [int(line.split(' ')[i]) for i in range(7)]
    exist_max_oid = max(exist_max_oid, oid)
    reid_hash_map[cameras[cid-min_cid]][(frame_id, oid)] = (left, top, width, height)


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

for cam in cameras:
    for time, id in reid_hash_map[cam].keys():
        if time not in reid_time_obj[cam]:
            reid_time_obj[cam][time] = set([id])
        else:
            reid_time_obj[cam][time].add(id)


def profile_camera_pair(source_cam, target_cam):
    TP, TN, FP, FN = 0, 0, 0, 0

    for frame_id in range(1, 1800):

        if frame_id - frame_diff[source_cam] < 50 or frame_id - frame_diff[source_cam] > 1500:
            continue

        target_frame_id = frame_id - frame_diff[source_cam] + frame_diff[target_cam]

        if  target_frame_id < 1:
            continue

        if frame_id not in reid_time_obj[source_cam]: 
            continue

        for reid_obj in reid_time_obj[source_cam][frame_id]:

            source_reid_rect = reid_hash_map[source_cam][(frame_id, reid_obj)]
            if frame_id not in gt_time_obj[source_cam]:
                TN += 1
                continue
            
            best_roi, best_gt_id, possible_gt_id = 0, -1, []
            for gt_obj in gt_time_obj[source_cam][frame_id]:
                source_gt_rect = gt_hash_map[source_cam][(frame_id, gt_obj)]
                current_roi = general.IoU(source_reid_rect, source_gt_rect)
                if  current_roi > 0.3:
                    possible_gt_id.append(gt_obj)
                    if current_roi > best_roi:
                        best_roi = current_roi
                        best_gt_id = gt_obj

            if best_gt_id == -1: 
                TN += 1
                continue


            if (target_frame_id, reid_obj) not in reid_hash_map[target_cam]:
                if (target_frame_id, best_gt_id) not in gt_hash_map[target_cam]:
                    TN += 1
                else:
                    FN += 1
            else:
                if (target_frame_id, best_gt_id) not in gt_hash_map[target_cam]:
                    FP += 1
                else:
                    target_reid_rect = reid_hash_map[target_cam][(target_frame_id, reid_obj)]
                    target_gt_rect = gt_hash_map[target_cam][(target_frame_id, best_gt_id)]
                    if general.IoU(target_gt_rect, target_reid_rect) > 0.3:
                        TP += 1
                    else:
                        FP += 1

    return TP, FP, FN, TN
    

if __name__ == '__main__':
    assert((77, 1590) in reid_hash_map['c003'])
    for source in cameras:
        for target in cameras:
            if source == target: continue
            # if not (source == 'c003' and target == 'c001'): continue
            print(source, target, profile_camera_pair(source, target))