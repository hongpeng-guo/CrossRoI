import os, cv2
import numpy as np
import optimizer
import visualizer

# environment & macro definations
FRAME_RATE = 10
WORKSAPCE = "/home/hongpeng/Desktop/research/AICity/Track3/"
DATA_PATH = "train/S01/" 
TIMESTAMP_PATH = "cam_timestamp/S01.txt"
GT_PATH = "gt/gt.txt"
DET_PATH = {"yolo3": "det/det_yolo3.txt", 
			"rcnn": "det/det_mask_rcnn.txt",
			"ssd": "det/det_ssd512.txt"}
TIMESTAMP_CONTENT = open(WORKSAPCE + TIMESTAMP_PATH).read().split('\n')[:-1]
ts_base = {each.split(' ')[0]: float(each.split(' ')[1]) for each in TIMESTAMP_CONTENT}

# get the k-th frame of a certain camera
def get_frame(cam_name, frame_id):
	cap = cv2.VideoCapture( WORKSAPCE + DATA_PATH + cam_name + '/' + 'vdo.avi')
	cap.set(1, frame_id)
	_, frame = cap.read()
	return frame

cameras = ["c001", "c002", "c003", "c004", "c005"]
# cameras = ["c001", "c002", "c003", "c004"]

cameras_shape = {camera: get_frame(camera, 0).shape[:2] for camera in cameras}
tile_height, tile_width = 40, 40
# tile_height, tile_width = 120, 120
cam_to_tshape = {cam: (cameras_shape[cam][0] // tile_height, cameras_shape[cam][1] // tile_width) for cam in cameras}

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

# convert bbox into 120 x 120 tiles 
def bbox_to_tiles(bbox, cam_name):
	f_height, f_width = cameras_shape[cam_name]
	row_n =  f_width // tile_width
	left, top, width, height = bbox
	first_tile = (top // tile_height) * row_n + (left // tile_width)
	last_tile = ((top + height) // tile_height) * row_n + ((left + width) // tile_width)
	result = []
	for tile in range(first_tile, last_tile + 1):
		if not (left // tile_width) <= tile % row_n <= ((left + width) // tile_width):
			continue
		result.append(tile)
	return result

# covernt gt results to camera data hashmap
def data_to_hashmap(cam_name, detector):
	gt = open(WORKSAPCE + DATA_PATH + cam_name + '/' + GT_PATH).read().split('\n')[:-1]
	det = open(WORKSAPCE + DATA_PATH + cam_name + '/' + DET_PATH[detector]).read().split('\n')[:-1]
	# print("gt length is " + str(len(gt)))

	gt_hashmap = {(int(each.split(',')[0]), int(each.split(',')[1])): \
				   tuple([int(i) for i in each.split(',')[2:7]]) for each in gt}
	det_hashmap = {}
	hashmap_len = 0
	for line in det:
		frame, _, left, top, width, height, confidence, _, _, _ = [float(each) for each in line.split(',')]
		frame, left, top, width, height = round(frame), round(left), round(top), round(width), round(height)
		if confidence < 0.5:
			continue
		if frame not in det_hashmap:
			det_hashmap[frame] = [(left, top, width, height, confidence)]
		else:
			det_hashmap[frame].append((left, top, width, height, confidence))
		hashmap_len += 1
	# print("det length is " + str(hashmap_len))
	
	for key in gt_hashmap:
		rect1 = gt_hashmap[key][:4]
		IoU_max, confidence = 0.0, 0.5
		if key[0] not in det_hashmap:
			gt_hashmap[key] = rect1 + (confidence,)
			continue
		for obj in det_hashmap[key[0]]:
			rect2 = obj[:4]
			if IoU(rect1, rect2) > IoU_max:
				IoU_max, confidence = IoU(rect1, rect2), obj[4]
		gt_hashmap[key] = rect1 + (confidence,)

	return gt_hashmap

# generate a hashmap containing all cam data in the 
# form of {(frame, obj_id): (tile_list, confidence}}
def multi_cam_hashmap(cam_list, detector, time_window):
	gt_multi_hashmap = {cam: data_to_hashmap(cam, detector) for cam in cam_list}
	frame_diff = {cam: int(FRAME_RATE * ts_base[cam]) for cam in cam_list}
	frame_diff = {cam: frame_diff[cam] - max(frame_diff.values()) for cam in cam_list}
	# convert bbox into tile list formation
	muti_sync_hashmap = {}
	time_to_obj = {t: set() for t in range(time_window[0], time_window[1])}
	for cam in gt_multi_hashmap:
		tmp_dic = {}
		for t, obj in gt_multi_hashmap[cam]:
			if t - frame_diff[cam] < time_window[0] or t - frame_diff[cam] >= time_window[1]:
				continue
			bbox, confidence = gt_multi_hashmap[cam][(t, obj)][:4], gt_multi_hashmap[cam][(t, obj)][4]
			tmp_dic[(t - frame_diff[cam], obj)] = (bbox_to_tiles(bbox, cam), confidence)
			time_to_obj[t - frame_diff[cam]].add(obj)
		muti_sync_hashmap[cam] = tmp_dic
	return muti_sync_hashmap, time_to_obj

if __name__ == "__main__":
	multi_used_tiles = optimizer.optimization_solver(cameras, cam_to_tshape, 'rcnn', [0, 600])

	multi_hashmap, time_to_obj = multi_cam_hashmap(cameras, 'rcnn', [0, 600])
	multi_bg_tiles = {cam: set(list(range(cam_to_tshape[cam][0] * cam_to_tshape[cam][1]))) for cam in cameras}
	for cam in cameras:
		used = [each[0] for each in multi_hashmap[cam].values()]
		for each in used:
			for item in each:
				if item in multi_bg_tiles[cam]:
					multi_bg_tiles[cam].remove(item)

	multi_nouse_tiles = {cam: set(list(range(cam_to_tshape[cam][0] * cam_to_tshape[cam][1]))) - set(multi_used_tiles[cam]) for cam in cameras}
	print([len(each) for each in multi_nouse_tiles.values() ])
	for cam in cameras:
		visualizer.plot_frame_w_nouse_tiles(cam, 10, multi_nouse_tiles[cam])

	# for cam_name in cameras:
	# 	base_frame = get_frame(cam_name, 10)
	# 	cv2.imwrite(cam_name + "_base.png", base_frame)


	# cameras = ["c001", "c002", "c003"]
	# multi_sync_hashmap, time_to_obj = multi_cam_hashmap(cameras, 'rcnn', [0, 600])
	# for t in range(10, 600):
	# 	print(t)
	# 	for obj in time_to_obj[t]:
	# 		# print(obj)
	# 		find = 3
	# 		for cam in  cameras:
	# 			if (t, obj) in multi_sync_hashmap[cam]:
	# 				find -= 1
	# 		if find > 1:
	# 			find = 3
	# 			continue
	# 		for cam in cameras:
	# 			base_frame = get_frame(cam, t)
	# 			cv2.imwrite(cam + "_reid.png", base_frame)
	# 		exit(0)

			
				