import math, cv2
from os import confstr_names
import numpy as np
from numpy.lib.npyio import _savez_compressed_dispatcher
import Optimizer
import Visualizer
import MergeTile
import SVMFilter
import RegressionFilter
import CreateGraph as cg
import Visualizer as vis
import matplotlib.pyplot as plt
from collections import OrderedDict


# environment & macro definations
FRAME_RATE = 10
WORKSAPCE = "/home/hongpeng/Desktop/research/AICity/Track3/"
DATA_PATH = "train/S01/" 
TIMESTAMP_PATH = "cam_timestamp/S01.txt"
GT_PATH = "gt/gt.txt"
TRACK_PATH = "mtsc/mtsc_deepsort_yolo3.txt"
DET_DIR = "../tensorflow-yolov3"
TIMESTAMP_CONTENT = open(WORKSAPCE + TIMESTAMP_PATH).read().split('\n')[:-1]
ts_base = {each.split(' ')[0]: float(each.split(' ')[1]) for each in TIMESTAMP_CONTENT}

cameras = ["c001", "c002", "c003", "c004", "c005"]
frame_diff = {cam: int(FRAME_RATE * ts_base[cam]) for cam in cameras}
frame_diff = {cam: frame_diff[cam] - max(frame_diff.values()) for cam in cameras}


def IoU(rect1,rect2):
	x1, y1, w1, h1 = rect1
	x2, y2, w2, h2 = rect2

	inter_w = (w1 + w2) - (max(x1 + w1, x2 + w2) - min(x1, x2))
	inter_h = (h1 + h2) - (max(y1 + h1, y2 + h2) - min(y1, y2))

	if inter_h<=0 or inter_w <= 0:
		return 0
	inter = inter_w * inter_h
	union = w1 * h1 + w2 * h2 - inter
	return inter / union


def time_to_det_bbox(cam_name):
    result = {}
    for line in open(DET_DIR + '/' + 'det_' + cam_name + '.txt', 'r').readlines():
        frame_id, left, top, right, buttom, confidence = [float(each) for each in line.split(' ')]
        frame_id = round(frame_id)

        if frame_id + 1 not in result:
            result[frame_id + 1] = [(left, top, right - left, buttom - top)]
        else:
            result[frame_id + 1].append((left, top, right - left, buttom - top))
    return result


def time_to_baseline_bbox(cam_name):
    print(cg.cameras_shape[cam_name][1])
    result = {}
    for line in open(DET_DIR + '/' + 'baseline_' + cam_name + '.txt', 'r').readlines():
        frame_id, left, top, right, buttom, confidence = [float(each) for each in line.split(' ')]
        frame_id = round(frame_id)

        # remove low confidence object 
        if confidence < 0.7: continue
        # remove margin none complete object
        if left < 150 or right + 150 > cg.cameras_shape[cam_name][1] or \
            top < 150 or buttom + 150 > cg.cameras_shape[cam_name][0]: continue
        # remove too small object from
        # if (right - left) * (buttom - top) < 800: continue

        if frame_id + 1  not in result:
            result[frame_id + 1] = [(left, top, right - left, buttom - top)]
        else:
            result[frame_id + 1].append((left, top, right - left, buttom - top))
    return result


def time_to_didi_reid_bbox_obj(cam_name):
    result = {}
    for line in open(WORKSAPCE +  'track1.txt', 'r').readlines():
        camera, obj_id, frame_id, left, top, width, height, _, _ = [round(int(each)) for each in line.split(' ')]
        if camera > 5: break
        if camera != int(cam_name[-1]): continue
        if frame_id  not in result:
            result[frame_id] = [(left, top, width, height, obj_id)]
        else:
            result[frame_id].append((left, top, width, height, obj_id))
    return result


def time_to_gt_bbox_obj(cam_name):
    result = {}
    for line in open(WORKSAPCE + DATA_PATH + cam_name + '/' + GT_PATH, 'r').readlines():
        frame_id, obj_id, left, top, width, height, _, _, _, _ = [round(int(each)) for each in line.split(',')]
        if frame_id  not in result:
            result[frame_id] = [(left, top, width, height, obj_id)]
        else:
            result[frame_id].append((left, top, width, height, obj_id))
    return result


def gt_time_bbox_obbj_Kalman_filter(cam_name, input_dict):
    result = {}
    obj_to_time_bbox = {}
    for frame_id in input_dict:
        for rect_obj in input_dict[frame_id]:
            rect, obj_id = rect_obj[:4], rect_obj[-1]
            if obj_id not in obj_to_time_bbox:
                obj_to_time_bbox[obj_id] = {frame_id: rect}
            else:
                obj_to_time_bbox[obj_id][frame_id] = rect

    for obj_id in obj_to_time_bbox:
        duration_map = OrderedDict(sorted(obj_to_time_bbox[obj_id].items()))
        exist_frames = list(duration_map.keys())
        for i, frame_id in enumerate(exist_frames):
            if i == 0: continue
            if exist_frames[i-1] + 1 == frame_id: continue
            for add_frame_id in range(exist_frames[i-1] + 1, frame_id):
                previous_value = tuple([each * (frame_id - add_frame_id) for each in duration_map[exist_frames[i-1]]])
                future_value = tuple([each * (add_frame_id - exist_frames[i-1]) for each in duration_map[frame_id]])
                l, t, w, h = tuple([(previous_value[j] + future_value[j]) // (frame_id - exist_frames[i-1]) for j in range(4)])
                l, t, w, h = max(l, 0), max(t, 0), min(cg.cameras_shape[cam_name][1]-l, w), min(cg.cameras_shape[cam_name][0]-t, h)
                obj_to_time_bbox[obj_id][add_frame_id] = (l, t, w, h)
        
        if len(exist_frames) == 1: continue

        begin = max(0, exist_frames[0] - 20)
        for add_frame_id in range(begin, exist_frames[0]):
            near_value = tuple([each * (exist_frames[1] - add_frame_id) for each in duration_map[exist_frames[0]]])
            far_value = tuple([each * (exist_frames[0] - add_frame_id) for each in duration_map[exist_frames[1]]])
            l, t, w, h = ((near_value[i] - far_value[i])  for i in range(4))
            l, t, w, h = max(l, 0), max(t, 0), min(cg.cameras_shape[cam_name][1]-l, w), min(cg.cameras_shape[cam_name][0]-t, h)
            obj_to_time_bbox[obj_id][add_frame_id] = (l, t, w, h)
 
        end = exist_frames[-1] + 20
        for add_frame_id in range(exist_frames[-1], end + 1):
            near_value = tuple([each * (-exist_frames[-2] + add_frame_id) for each in duration_map[exist_frames[-1]]])
            far_value = tuple([each * (-exist_frames[-1] + add_frame_id) for each in duration_map[exist_frames[-2]]])
            l, t, w, h = ((near_value[i] - far_value[i])  for i in range(4))
            l, t, w, h = max(l, 0), max(t, 0), min(cg.cameras_shape[cam_name][1]-l, w), min(cg.cameras_shape[cam_name][0]-t, h)
            obj_to_time_bbox[obj_id][add_frame_id] = (l, t, w, h)
 
    for obj_id in obj_to_time_bbox:
        for frame_id in obj_to_time_bbox[obj_id]:
            rect = obj_to_time_bbox[obj_id][frame_id]
            if frame_id not in result:
                result[frame_id] = [rect + (obj_id,)]
            else:
                result[frame_id].append(rect + (obj_id,))
    return result



def time_to_track_bbox_obj(cam_name):
    result = {}

    track_id_time_to_bbox = {}
    for line in open(WORKSAPCE + DATA_PATH + cam_name + '/' + TRACK_PATH, 'r').readlines():
        frame_id, obj_id, left, top, width, height, _, _, _, _ = [round(float(each)) for each in line.split(',')]
        if obj_id  not in track_id_time_to_bbox:
            track_id_time_to_bbox[obj_id] = {frame_id: (left, top, width, height)}
        else:
            track_id_time_to_bbox[obj_id][frame_id] = (left, top, width, height)

    total_obj = len(track_id_time_to_bbox)
    re_id_obj_count = 0

    gt_time_to_bbox_obj = time_to_gt_bbox_obj(cam_name)
    for obj_id in track_id_time_to_bbox:
        reid_candidates = {}
        for frame_id in track_id_time_to_bbox[obj_id]:
            if frame_id not in gt_time_to_bbox_obj: 
                continue
            for gt_bbox_obj in gt_time_to_bbox_obj[frame_id]:
                track_rect = track_id_time_to_bbox[obj_id][frame_id]
                gt_bbox, gt_obj = gt_bbox_obj[:4], gt_bbox_obj[-1]
                IoU_score = IoU(track_rect, gt_bbox)
                if IoU_score > 0.3:
                    if gt_obj not in reid_candidates:
                        reid_candidates[gt_obj] = IoU_score
                    else:
                        reid_candidates[gt_obj] += IoU_score

        if len(reid_candidates) == 0:
            new_obj_id = -1
        else:
            new_obj_id, obj_score = max(reid_candidates, key=reid_candidates.get), max(reid_candidates.keys())
            if obj_score < 0.8:
                new_obj_id = -1
            else:
                re_id_obj_count += 1

        for frame_id in track_id_time_to_bbox[obj_id]:
            bbox = track_id_time_to_bbox[obj_id][frame_id]
            if frame_id not in result:
                result[frame_id] = [bbox + (new_obj_id, )]
            else:
                result[frame_id].append(bbox + (new_obj_id, ))
    
    print(cam_name, total_obj, re_id_obj_count)
    return result




def compare_baseline_det_bbox(cam_name):
    det_t_to_bbox = time_to_det_bbox(cam_name)
    baseline_t_to_bbox = time_to_baseline_bbox(cam_name)
    time = baseline_t_to_bbox.keys()
    same_result, diff_result = {t: [] for t in time}, {t: [] for t in time}
    for t in time:
        if t not in det_t_to_bbox:
            diff_result[t] = baseline_t_to_bbox[t]
            continue
        for baseline_rect in baseline_t_to_bbox[t]:
            IoU_score = 0
            for det_rect in det_t_to_bbox[t]:
                IoU_score = max(IoU_score, IoU(baseline_rect, det_rect))
            if IoU_score < 0.3:
                diff_result[t].append(baseline_rect)
            else:
                same_result[t].append(baseline_rect)
    return same_result, diff_result


def assign_ID_time_to_bbox(cam_name):
    same_result, diff_result = compare_baseline_det_bbox(cam_name)
    gt_time_to_obj_id = time_to_gt_bbox_obj(cam_name)
    gt_time_to_obj_id = gt_time_bbox_obbj_Kalman_filter(cam_name, gt_time_to_obj_id)

    def assign_id(id_dict, target_dict):
        time = target_dict.keys()
        for t in time:
            if t not in id_dict:
                bboxes = target_dict[t]
                id_bboxes = [each + (-1,) for each in bboxes]
                target_dict[t] = id_bboxes
                continue
            for idx, target_rect in enumerate(target_dict[t]):
                best_fit_id, IoU_score = -1, 0
                for gt_rect_id in id_dict[t]:
                    gt_rect, id = gt_rect_id[:4], gt_rect_id[-1]
                    IoU_tmp = IoU(target_rect, gt_rect)
                    if IoU_tmp < 0.2: continue
                    if IoU_tmp > IoU_score:
                        IoU_score = IoU_tmp
                        best_fit_id = id
                target_dict[t][idx] = target_dict[t][idx] + (best_fit_id,)
        return target_dict

    return assign_id(gt_time_to_obj_id, same_result),\
           assign_id(gt_time_to_obj_id, diff_result)


if __name__ == '__main__':
    cameras_same_t_to_bbox_id = {cam: {} for cam in cameras}
    cameras_diff_t_to_bbox_id = {cam: {} for cam in cameras}

    for cam_name in cameras:
        same_t_to_bbox_id, diff_t_to_bbox_id = assign_ID_time_to_bbox(cam_name)
        cameras_same_t_to_bbox_id[cam_name] = same_t_to_bbox_id
        cameras_diff_t_to_bbox_id[cam_name] = diff_t_to_bbox_id

    time_window = [600, 1500]

    baseline_result = [0 for _ in range(time_window[1] - time_window[0])]
    det_result = [0 for _ in range(time_window[1] - time_window[0])]

    evaluate_interval, interval_shared_stack = 5, []

    for t in range(time_window[0], time_window[1]):
        unique_count, shared_set = 0, set()
        for cam_name in cameras:
            real_t = t - frame_diff[cam_name]
            if real_t not in cameras_same_t_to_bbox_id[cam_name]: continue
            for _, _, _, _, id in cameras_same_t_to_bbox_id[cam_name][real_t]:
                if id == -1: 
                    unique_count += 1
                else:
                    shared_set.add(id)
        det_result[t - time_window[0]] = unique_count + len(shared_set)

        if len(interval_shared_stack) == evaluate_interval:
            interval_shared_stack.pop(0)
        interval_shared_stack.append(shared_set)

        diff_unique_count = 0
        for cam_name in cameras:
            real_t = t - frame_diff[cam_name]
            if real_t not in cameras_diff_t_to_bbox_id[cam_name]: continue
            for left, top, width, height, id in cameras_diff_t_to_bbox_id[cam_name][real_t]:
                baseframe = cg.get_frame(cam_name, real_t - 1)
                left, top, width, height = int(left), int(top), int(width), int(height)
                print(left, top, left+width, top+height)
                if id == -1: 
                    diff_unique_count += 1
                    continue
                elif id not in set.union(* interval_shared_stack):
                    diff_unique_count += 1
                    shared_set.add(id)
                    continue
                else:
                    continue
                baseframe = cv2.cv2.rectangle(baseframe, (left, top), (left + width, top + height), (255, 0, 0), 3)
                cv2.imshow('image', baseframe)
                cv2.waitKey()
                cv2.destroyAllWindows()
        baseline_result[t - time_window[0]] = diff_unique_count + det_result[t - time_window[0]]

    diff = [max( baseline_result[i] - det_result[i], 0 ) for i in range(len(baseline_result))]

    print(sum(diff), sum(baseline_result), sum(diff)/ sum(baseline_result))
    plt.plot(baseline_result)
    plt.plot(diff)
    plt.show()

    for cam_name in cameras:
        det_time_to_bbox = cameras_same_t_to_bbox_id[cam_name]
        bbox_count = []
        for i in range(1, 1800):
            if i not in det_time_to_bbox:
                bbox_count.append(0)
            else:
                bbox_count.append(len(det_time_to_bbox[i]))

        plt.plot(bbox_count)
        plt.show()