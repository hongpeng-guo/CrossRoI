import math, cv2
import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt
import General as general

# environment & macro definations
FRAME_RATE = general.FRAME_RATE
WORKSAPCE = general.WORKSAPCE
DATA_PATH = general.DATA_PATH
DET_PATH =  general.DET_PATH
ts_base = general.ts_base
cameras = general.cameras
cameras_shape = general.cameras_shape


reid_hash_map = {cam: {} for cam in cameras}
exist_max_oid = -1

## covernt gt results to camera data hashmap
#
def data_to_hashmap(cam_name_path):
    gt = open(cam_name_path).read().split('\n')[:-1]
    gt_hashmap = {}
    for each in gt:
        if int(each.split(',')[0]) not in gt_hashmap:
            gt_hashmap[int(each.split(',')[0])] = [tuple([int(i) for i in each.split(',')[2:6]])]
        else:
            gt_hashmap[int(each.split(',')[0])].append(tuple([int(i) for i in each.split(',')[2:6]]))
    return gt_hashmap

for cam in cameras:
    reid_hash_map[cam] = data_to_hashmap("/home/hongpeng/Desktop/research/AICity/Track3/train/S01/" + cam + '/' + 'gt/gt.txt')


BASELINE_DET_DIR = "../tensorflow-yolov3/results/" + general.SCENE_NAME + "/baseline"
def time_to_baseline_bbox(cam_name):
    result = {}
    for line in open(BASELINE_DET_DIR + '/' + 'det_' + cam_name + '.txt', 'r').readlines():
        frame_id, left, top, right, buttom, confidence = [float(each) for each in line.split(' ')]
        frame_id = round(frame_id)

        # remove low confidence object 
        if confidence < 0.2: continue
        # remove margin none complete object
        if left < 50 or right + 50 > general.cameras_shape[cam_name][1] or \
            top < 50 or buttom + 50 > general.cameras_shape[cam_name][0]: continue
        # remove too small object from
        # if (right - left) * (buttom - top) < 800: continue

        if frame_id + 1  not in result:
            result[frame_id + 1] = [(left, top, right - left, buttom - top)]
        else:
            result[frame_id + 1].append((left, top, right - left, buttom - top))
    return result


def plot_frame_bboxes(cam_name, frame_id):
    gt_detection = reid_hash_map[cam_name]
    yolo_detection = time_to_baseline_bbox(cam_name)

    gt_rects = gt_detection[frame_id]
    ab_rects, single_rects = [], []
    for yolo_rect in yolo_detection[frame_id]:
        max_iou = 0
        for gt_rect in gt_rects:
            max_iou = max(max_iou, general.IoU(gt_rect, yolo_rect)) 
        if max_iou == 0:
            single_rects.append(yolo_rect)
        elif max_iou < 0.2:
            ab_rects.append(yolo_rect)

    if not (len(ab_rects) >=1 and len(single_rects) >= 3): return

    base_frame = general.get_frame(cam_name, frame_id-1)
    for rect in gt_rects:
        left, top, width, height = rect
        cv2.rectangle(base_frame, (left, top), (left + width, top + height), (255, 0, 0), 3)
    for rect in ab_rects:
        left, top, width, height = [int(each) for each in rect]
        cv2.rectangle(base_frame, (left, top), (left + width, top + height), (0, 255, 0), 3)
    for rect in single_rects:
        left, top, width, height = [int(each) for each in rect]
        cv2.rectangle(base_frame, (left, top), (left + width, top + height), (0, 0, 255), 3)

    print(frame_id)
    cv2.imshow('image', base_frame)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # cv2.imwrite('figures/appendix.png', base_frame)

for i in range(800, 1000):
    plot_frame_bboxes('c002', i)