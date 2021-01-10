import math, cv2
from time import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
from matplotlib import pyplot as plt

# environment & macro definations
FRAME_RATE = 10
WORKSAPCE = "/home/hongpeng/Desktop/research/AICity/Track3/"
DATA_PATH = "train/S01/"
TIMESTAMP_PATH = "/home/hongpeng/Desktop/research/AICity/Track3/cam_timestamp/S01.txt"
REID_PATH = "reid.txt"
DET_PATH =  "det/det_yolo3.txt"
TIMESTAMP_CONTENT = open(TIMESTAMP_PATH).read().split('\n')[:-1]
ts_base = {each.split(' ')[0]: float(each.split(' ')[1]) for each in TIMESTAMP_CONTENT}

cameras = ["c001", "c002", "c003", "c004", "c005"]

# get the k-th frame of a certain camera
def get_frame(cam_name, frame_id):
	cap = cv2.VideoCapture( WORKSAPCE + DATA_PATH + cam_name + '/' + 'vdo.avi')
	cap.set(1, frame_id)
	_, frame = cap.read()
	return frame

cameras_shape = {camera: get_frame(camera, 0).shape[:2] for camera in cameras}

# IoU score of two given rectanglar bbox
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

reid_hash_map = {cam: {} for cam in cameras}
exist_max_oid = 0

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
    if int(line.split(' ')[0]) > 5: 
        break
    cid, oid, frame_id, left, top, width, height = [int(line.split(' ')[i]) for i in range(7)]
    exist_max_oid = max(exist_max_oid, oid)
    reid_hash_map[cameras[cid-1]][(frame_id, oid)] = (left, top, width, height)

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
                if IoU(exist_bbox, (left, top, width, height)) > 0.1:
                    add_this = False
                    break
        if add_this:
            exist_max_oid += 1
            reid_hash_map[cam_name][(frame_id, exist_max_oid)] = (left, top, width, height)



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

    return np.array(svm_data), np.array(svm_label), data_id


outlier_dict = {cam: [] for cam in cameras}
unique_dict = {cam: [] for cam in cameras}

for source_cam in cameras:
    source_unique = set(reid_hash_map[source_cam].keys())
    source_outlier = set()
    for destination_cam in cameras:
        if source_cam == destination_cam:
            continue
        svm_data, svm_label, data_id = prepare_svm_data(reid_hash_map, source_cam, destination_cam, [0, 900])       
        false_pos = list(np.where(svm_label == 0)[0])
        source_unique = source_unique.intersection(set([data_id[i] for i in false_pos]))

        clf = svm.SVC(kernel='rbf', class_weight='balanced', gamma=2e-5)  # change to 2e-5 in overall setup
        clf.fit(svm_data, svm_label)
        y_pred = clf.predict(svm_data)

        false_negative_pos = list(np.where((svm_label - y_pred) == -1)[0])
        if len(false_negative_pos) == 0: continue
        tmp_outlier = set([data_id[i] for i in false_negative_pos])
        source_outlier = source_outlier.union(tmp_outlier)    
    outlier_dict[source_cam] = source_unique.intersection(source_outlier)
    unique_dict[source_cam] = source_unique

Multi_Hashmap = {}
for cam in cameras:
    for key in outlier_dict[cam]:
        del reid_hash_map[cam][key]
    Multi_Hashmap[cam] = reid_hash_map[cam]


if __name__ == "__main__":

    import CreateGraph as cg

    for cam in outlier_dict:
        print(len(outlier_dict[cam]))

    image_frame_id = {'c001': 630, 'c002': 630, 'c003': 630, 'c004': 630, 'c005': 630}

    unique_area = []
    total_area = []

    for cam in cameras:
        cam_shape =  (cg.cameras_shape[cam][1], cg.cameras_shape[cam][0], 3)
        base_frame = np.zeros(cam_shape).astype(np.uint8)
        dark_frame = np.zeros(cam_shape).astype(np.uint8)
        for key in unique_dict[cam] - outlier_dict[cam] :
            left, top, width, height = reid_hash_map[cam][key]
            base_frame[left: left+width, top: top+height] = 255

        unique_area.append(np.sum(base_frame) / (255 * 3))
        total_area.append(cam_shape[0] * cam_shape[1])

        base_frame = np.moveaxis(base_frame, 0, 1)
        dark_frame = np.moveaxis(dark_frame, 0, 1)

        image_frame = cg.get_frame(cam, image_frame_id[cam])
        print(image_frame.shape, dark_frame.shape)
        combine_frame = cv2.addWeighted( dark_frame, 0.8, image_frame, 0.5, 1)

        transparent_loc = tuple(np.where(base_frame == 255))
        combine_frame[transparent_loc] = image_frame[transparent_loc]

        print(unique_area[-1] / total_area[-1])

        cv2.imshow('image', combine_frame)
        cv2.waitKey()
        cv2.destroyAllWindows()

    print(sum(unique_area) / sum(total_area))