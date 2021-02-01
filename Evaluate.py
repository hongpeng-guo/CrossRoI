import math, cv2
from os import confstr_names
import numpy as np
from numpy.lib.npyio import _savez_compressed_dispatcher
import General as general
import matplotlib.pyplot as plt
from collections import OrderedDict
import Reducto as reducto
from reducto.differencer import AreaDiff


# environment & macro definations
FRAME_RATE = general.FRAME_RATE
WORKSAPCE = general.WORKSAPCE
DATA_PATH = general.DATA_PATH
GT_PATH = general.GT_PATH
TRACK_PATH = general.TRACK_PATH
ts_base = general.ts_base
cameras = general.cameras
frame_diff = general.frame_diff
experiment_subdirs = general.experiment_subdirs

# detection environment from my sbnet-yolov3
BASELINE_DET_DIR = "../tensorflow-yolov3/results/" + general.SCENE_NAME + "/baseline"
CROP_DET_DIR = "../tensorflow-yolov3/results/" + general.SCENE_NAME 


def time_to_det_bbox(cam_name):
    result = {}
    for line in open(CROP_DET_DIR + '/' + 'det_' + cam_name + '.txt', 'r').readlines():
        frame_id, left, top, right, buttom, confidence = [float(each) for each in line.split(' ')]
        frame_id = round(frame_id)

        if frame_id + 1 not in result:
            result[frame_id + 1] = [(left, top, right - left, buttom - top)]
        else:
            result[frame_id + 1].append((left, top, right - left, buttom - top))
    return result


def time_to_baseline_bbox(cam_name):
    # print(general.cameras_shape[cam_name][1])
    result = {}
    for line in open(BASELINE_DET_DIR + '/' + 'det_' + cam_name + '.txt', 'r').readlines():
        frame_id, left, top, right, buttom, confidence = [float(each) for each in line.split(' ')]
        frame_id = round(frame_id)

        # remove low confidence object 
        if confidence < 0.7: continue
        # remove margin none complete object
        if left < 100 or right + 100 > general.cameras_shape[cam_name][1] or \
            top < 100 or buttom + 100 > general.cameras_shape[cam_name][0]: continue
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


def gt_time_bbox_obj_Kalman_filter(cam_name, input_dict):
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
                l, t, w, h = max(l, 0), max(t, 0), min(general.cameras_shape[cam_name][1]-l, w), min(general.cameras_shape[cam_name][0]-t, h)
                obj_to_time_bbox[obj_id][add_frame_id] = (l, t, w, h)
        
        if len(exist_frames) == 1: continue

        begin = max(0, exist_frames[0] - 20)
        for add_frame_id in range(begin, exist_frames[0]):
            near_value = tuple([each * (exist_frames[1] - add_frame_id) for each in duration_map[exist_frames[0]]])
            far_value = tuple([each * (exist_frames[0] - add_frame_id) for each in duration_map[exist_frames[1]]])
            l, t, w, h = ((near_value[i] - far_value[i])  for i in range(4))
            l, t, w, h = max(l, 0), max(t, 0), min(general.cameras_shape[cam_name][1]-l, w), min(general.cameras_shape[cam_name][0]-t, h)
            obj_to_time_bbox[obj_id][add_frame_id] = (l, t, w, h)
 
        end = exist_frames[-1] + 20
        for add_frame_id in range(exist_frames[-1], end + 1):
            near_value = tuple([each * (-exist_frames[-2] + add_frame_id) for each in duration_map[exist_frames[-1]]])
            far_value = tuple([each * (-exist_frames[-1] + add_frame_id) for each in duration_map[exist_frames[-2]]])
            l, t, w, h = ((near_value[i] - far_value[i])  for i in range(4))
            l, t, w, h = max(l, 0), max(t, 0), min(general.cameras_shape[cam_name][1]-l, w), min(general.cameras_shape[cam_name][0]-t, h)
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
                IoU_score = general.IoU(track_rect, gt_bbox)
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




def compare_baseline_det_bbox(cam_name, mask_data=None):
    # det_t_to_bbox = time_to_det_bbox(cam_name)
    baseline_t_to_bbox = time_to_baseline_bbox(cam_name)
    time = baseline_t_to_bbox.keys()
    same_result, diff_result = {t: [] for t in time}, {t: [] for t in time}
    for t in time:
        # if t not in det_t_to_bbox:
        #     diff_result[t] = baseline_t_to_bbox[t]
        #     continue
        for baseline_rect in baseline_t_to_bbox[t]:
            left, top, width, height =  [round(each) for each in baseline_rect]
            IoU_score = np.sum(mask_data[top:top+height, left:left+width]) / (width * height)
            # for det_rect in det_t_to_bbox[t]:
            #     IoU_score = max(IoU_score, general.IoU(baseline_rect, det_rect))
            if IoU_score < 0.5:
                diff_result[t].append(baseline_rect)
            else:
                same_result[t].append(baseline_rect)
    return same_result, diff_result


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
                IoU_tmp = general.IoU(target_rect, gt_rect)
                if IoU_tmp < 0.1: continue
                if IoU_tmp > IoU_score:
                    IoU_score = IoU_tmp
                    best_fit_id = id
            target_dict[t][idx] = target_dict[t][idx] + (best_fit_id,)
    return target_dict

def assign_ID_time_to_bbox(cam_name, mask_data=None):
    same_result, diff_result = compare_baseline_det_bbox(cam_name, mask_data)
    gt_time_to_obj_id = time_to_gt_bbox_obj(cam_name)
    gt_time_to_obj_id = gt_time_bbox_obj_Kalman_filter(cam_name, gt_time_to_obj_id)

    return assign_id(gt_time_to_obj_id, same_result),\
           assign_id(gt_time_to_obj_id, diff_result)


def car_counting_Baseline_Roi(cameras, time_window, \
                              scene_name='S01', setting_name='2e-05_1.0'):

    cameras_same_t_to_bbox_id = {cam: {} for cam in cameras}
    cameras_diff_t_to_bbox_id = {cam: {} for cam in cameras}

    for cam_name in cameras:
        mask_path = 'videos/' + scene_name + '/' + setting_name + '/' + cam_name + '_mask.jpg'
        mask_data = cv2.imread(mask_path)[:,:,0] // 255
        same_t_to_bbox_id, diff_t_to_bbox_id = assign_ID_time_to_bbox(cam_name, mask_data)
        cameras_same_t_to_bbox_id[cam_name] = same_t_to_bbox_id
        cameras_diff_t_to_bbox_id[cam_name] = diff_t_to_bbox_id

    baseline_result = [0 for _ in range(time_window[1] - time_window[0])]
    det_result = [0 for _ in range(time_window[1] - time_window[0])]

    cache_interval, interval_shared_stack = 3, []

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

        if len(interval_shared_stack) == cache_interval:
            interval_shared_stack.pop(0)
        interval_shared_stack.append(shared_set)

        diff_unique_count = 0
        for cam_name in cameras:
            real_t = t - frame_diff[cam_name]
            if real_t not in cameras_diff_t_to_bbox_id[cam_name]: continue
            for left, top, width, height, id in cameras_diff_t_to_bbox_id[cam_name][real_t]:
                # baseframe = general.get_frame(cam_name, real_t - 1)
                # left, top, width, height = int(left), int(top), int(width), int(height)
                # print(left, top, left+width, top+height)
                if id == -1: 
                    diff_unique_count += 1
                    continue
                elif id not in set.union(* interval_shared_stack):
                    diff_unique_count += 1
                else:
                    continue

                # baseframe = cv2.cv2.rectangle(baseframe, (left, top), (left + width, top + height), (255, 0, 0), 3)
                # cv2.imshow('image', baseframe)
                # cv2.waitKey()
                # cv2.destroyAllWindows()

        baseline_result[t - time_window[0]] = diff_unique_count + det_result[t - time_window[0]]

    diff = [max( baseline_result[i] - det_result[i], 0 ) for i in range(len(baseline_result))]

    print(sum(diff), sum(baseline_result), sum(diff)/ sum(baseline_result))
    plt.plot(baseline_result)
    plt.plot(diff)
    plt.show()

    return baseline_result, det_result


# Note that selected_frames are 1 indexed.
def car_counting_Reducto(cameras, selected_frames, time_window):
    cameras_t_to_bbox_id = {cam: {} for cam in cameras}
    
    for cam_name in cameras:
        baseline_t_to_bbox = time_to_baseline_bbox(cam_name)
        gt_time_to_obj_id = time_to_gt_bbox_obj(cam_name)
        gt_time_to_obj_id = gt_time_bbox_obj_Kalman_filter(cam_name, gt_time_to_obj_id)
    
        cameras_t_to_bbox_id[cam_name] = assign_id(gt_time_to_obj_id, baseline_t_to_bbox)

    baseline_result = [0 for _ in range(time_window[1] - time_window[0])]

    evaluate_interval, interval_shared_stack = 3, []

    prev_det_frame = {cam_name: time_window[0] for cam_name in cameras}

    for t in range(time_window[0], time_window[1]):
        unique_count, shared_set = 0, set()
        for cam_name in cameras:
            if t + 1 in selected_frames[cam_name]:
                real_t = t - frame_diff[cam_name]
                prev_det_frame[cam_name] = t
            else:
                real_t = prev_det_frame[cam_name] - frame_diff[cam_name]

            if real_t not in cameras_t_to_bbox_id[cam_name]: continue
            for _, _, _, _, id in cameras_t_to_bbox_id[cam_name][real_t]:
                if id == -1: 
                    unique_count += 1
                else:
                    shared_set.add(id)
        baseline_result[t - time_window[0]] = unique_count + len(shared_set)

        if len(interval_shared_stack) == evaluate_interval:
            interval_shared_stack.pop(0)
        interval_shared_stack.append(shared_set)

    return baseline_result


# Note that selected_frames are 1 indexed.
def car_counting_ReductoRoi(cameras, selected_frames, time_window, \
                            scene_name='S01', setting_name='2e-05_1.0'):

    cameras_same_t_to_bbox_id = {cam: {} for cam in cameras}

    for cam_name in cameras:
        mask_path = 'videos/' + scene_name + '/' + setting_name + '/' + cam_name + '_mask.jpg'
        mask_data = cv2.imread(mask_path)[:,:,0] // 255
        same_t_to_bbox_id, _ = assign_ID_time_to_bbox(cam_name, mask_data)
        cameras_same_t_to_bbox_id[cam_name] = same_t_to_bbox_id

    det_result = [0 for _ in range(time_window[1] - time_window[0])]

    evaluate_interval, interval_shared_stack = 3, []

    prev_det_frame = {cam_name: time_window[0] for cam_name in cameras}

    for t in range(time_window[0], time_window[1]):
        unique_count, shared_set = 0, set()
        for cam_name in cameras:
            if t + 1 in selected_frames[cam_name]:
                real_t = t - frame_diff[cam_name]
                prev_det_frame[cam_name] = t
            else:
                real_t = prev_det_frame[cam_name] - frame_diff[cam_name]

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

    return det_result


# get selected frames in reducto retting.
def get_reducto_selected_frames(cameras_t_to_count, roi=False, \
                                scene_setting='S01', crop_setting='2e-05_1.0'):
    result = {}

    for cam_name in cameras_t_to_count:
        bbox_count_dict = {}
        for i in range(0, 1800):
            if i not in cameras_t_to_count[cam_name]:
                bbox_count_dict[i+1] = 0
            else:
                bbox_count_dict[i+1] = cameras_t_to_count[cam_name][i]
        if roi:
            video_path = 'videos/' + scene_setting + '/' + crop_setting + '/' + 'croped_' + cam_name + '.mp4'
        else:
            video_path = 'videos/' + scene_setting + '/' + 'baseline/' + 'h264_' + cam_name + '.mp4'

        diff_vectors = reducto.get_segmented_diff_vectors(video_path, segment_limit=90)
        train_vectors, test_vectors = diff_vectors[:30], diff_vectors[30:] 

        thresholds = [0.0, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]
        diff_results = reducto.get_segmented_diff_results(
                                train_vectors, 
                                thresholds)
        evaluations = reducto.get_segmented_evaluations(diff_results, bbox_count_dict)

        thresh_map = reducto.generate_hashmap(evaluations, train_vectors)

        selected_frames = reducto.generate_test_result(test_vectors, 600, bbox_count_dict, thresh_map)

        result[cam_name] = selected_frames

    return result
        

if __name__ == '__main__':
    CROP_DET_ROOT = CROP_DET_DIR

    for setting_name in experiment_subdirs:

        # if setting_name != '2e-05_0.1': continue

        print(setting_name)

        CROP_DET_DIR = CROP_DET_ROOT + '/' + setting_name

        cameras_same_t_to_bbox_id = {cam: {} for cam in cameras}
        cameras_diff_t_to_bbox_id = {cam: {} for cam in cameras}

        for cam_name in cameras:
            mask_path = 'videos/' + general.SCENE_NAME + '/' + setting_name + '/' + cam_name + '_mask.jpg'
            mask_data = cv2.imread(mask_path)[:,:,0] // 255
            same_t_to_bbox_id, diff_t_to_bbox_id = assign_ID_time_to_bbox(cam_name, mask_data)
            cameras_same_t_to_bbox_id[cam_name] = same_t_to_bbox_id
            cameras_diff_t_to_bbox_id[cam_name] = diff_t_to_bbox_id
        
        ## The following logic block works for Reducto Mudule
        # roi_t_to_count = { cam_name: { t: len(cameras_same_t_to_bbox_id[cam_name][t]) \
        #                             for t in cameras_same_t_to_bbox_id[cam_name]} \
        #                 for cam_name in cameras_same_t_to_bbox_id } 

        # baseline_t_to_count = { cam_name: { t: len(cameras_same_t_to_bbox_id[cam_name][t]) + roi_t_to_count[cam_name][t]\
        #                             for t in cameras_same_t_to_bbox_id[cam_name]} \
        #                 for cam_name in cameras_same_t_to_bbox_id } 

        # roi_selected_frames = get_reducto_selected_frames(roi_t_to_count, roi=True, \
        #                                                   scene_setting='S01', crop_setting=setting_name)

        # baseline_selected_frames = get_reducto_selected_frames(baseline_t_to_count, roi=False)

        # reducto_res = car_counting_Reducto(cameras, baseline_selected_frames, [600, 1800])

        # reductoroi_res = car_counting_ReductoRoi(cameras, roi_selected_frames, [600, 1800])

        baseline_res, roi_res = car_counting_Baseline_Roi(cameras, [300, 1300], \
                                                          scene_name=general.SCENE_NAME, setting_name=setting_name)

        # roi_err = [abs(baseline_res[i] - roi_res[i]) for i in range(len(baseline_res))]

        # reducto_err = [abs(baseline_res[i] - reducto_res[i]) for i in range(len(baseline_res))]
        # reductoroi_err = [abs(baseline_res[i] - reductoroi_res[i]) for i in range(len(baseline_res))]

        # print(setting_name, ' baseline: ',  sum(baseline_res) / len(baseline_res))
        # print(setting_name, ' roi_error: ', sum(roi_err) / sum(baseline_res))
        # print(sum(reducto_err) / sum(baseline_res))
        # print(sum(reductoroi_err) / sum(baseline_res))

        # plt.plot(roi_err)
        # plt.show()
        # plt.plot(reducto_err)
        # plt.show()
        # plt.plot(reductoroi_err)
        # plt.show()