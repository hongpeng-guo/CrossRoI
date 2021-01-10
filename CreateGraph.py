import math, cv2
from os import confstr_names
import numpy as np
from numpy.lib.npyio import _savez_compressed_dispatcher
import Optimizer
import Visualizer
import MergeTile
import SVMFilter
import RegressionFilter

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

cameras_shape = {camera: get_frame(camera, 0).shape[:2] for camera in cameras}
tile_height, tile_width = 64, 64
cam_to_tshape = {cam: (math.ceil(cameras_shape[cam][0] / tile_height), math.ceil(cameras_shape[cam][1] / tile_width)) for cam in cameras}
# print(cam_to_tshape)

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
def data_to_hashmap(cam_name):
	gt = open(WORKSAPCE + DATA_PATH + cam_name + '/' + GT_PATH).read().split('\n')[:-1]

	gt_hashmap = {(int(each.split(',')[0]), int(each.split(',')[1])): \
				   tuple([int(i) for i in each.split(',')[2:6]]) for each in gt}
	return gt_hashmap

# generate a hashmap containing all cam data in the 
# form of {(frame, obj_id): (tile_list, confidence}}
def multi_cam_hashmap(cam_list, time_window, gt_multi_hashmap=None):
	if gt_multi_hashmap is None:
		gt_multi_hashmap = {cam: data_to_hashmap(cam) for cam in cam_list}
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
			bbox = gt_multi_hashmap[cam][(t, obj)]
			tmp_dic[(t - frame_diff[cam], obj)] = (bbox_to_tiles(bbox, cam))
			time_to_obj[t - frame_diff[cam]].add(obj)
		muti_sync_hashmap[cam] = tmp_dic
	return muti_sync_hashmap, time_to_obj


# Generate ffmpeg scripts to crop video into tiles
def generate_ffmpeg_crop_scripts(cameras_to_tiles):
	for cam_idx, cam_name in enumerate(cameras_to_tiles):
		scripts = 'ffmpeg -re -i ' + WORKSAPCE + DATA_PATH + cam_name + '/' + 'vdo.avi' + ' -filter_complex '
		complex_variable = '\"[0]split={}'.format(len(cameras_to_tiles[cam_name]))
		for i in range(len(cameras_to_tiles[cam_name])):
			complex_variable += '[s{}]'.format(i)
		complex_variable += '; '
		for i, rec in enumerate(cameras_to_tiles[cam_name]):
			top, left, height, width = rec
			complex_variable += '[s{}]crop={}:{}:{}:{}[s{}]; '.format(i, width, height, left, top, i)
		complex_variable = complex_variable[:-2] + '\" '
		scripts += complex_variable
		for i in range(len(cameras_to_tiles[cam_name])):
			top, left, height, width = cameras_to_tiles[cam_name][i]
			scripts += '-map [s{}] -c:v libx264 videos/tmp/{}_{:02d}.avi '.format(i, cam_name, i)
		sh_file = open("scripts/ffmpeg_crop_{}.sh".format(cam_name), "w")
		sh_file.write(scripts)
		sh_file.close()


# Generate ffmpeg scripts to crop video into tiles
def generate_ffmpeg_merge_scripts(cameras_to_tiles):
	for cam_idx, cam_name in enumerate(cameras_to_tiles):
		scripts = 'ffmpeg '
		for i in range(len(cameras_to_tiles[cam_name])):
			scripts += '-i videos/tmp/{}_{:02d}.avi '.format(cam_name, i)
		scripts += '-filter_complex '

		complex_variable = '\"nullsrc=size={}x{} [base]; '.format(cameras_shape[cam_name][1], cameras_shape[cam_name][0])
		for i in range(len(cameras_to_tiles[cam_name])):
			complex_variable += '[{}:v] setpts=PTS-STARTPTS [s{}]; '.format(i, i)

		for i, rec in enumerate(cameras_to_tiles[cam_name]):
			top, left, _, _ = rec
			if len(cameras_to_tiles[cam_name]) == 1:
				complex_variable += '[base][s{}] overlay=shortest=1:x={}:y={} \" '.format(i, left, top)
				break
			if i == 0:
				complex_variable += '[base][s{}] overlay=shortest=1:x={}:y={} [tmp{}]; '.format(i, left, top, i)
			elif i < len(cameras_to_tiles[cam_name]) - 1:
				complex_variable += '[tmp{}][s{}] overlay=shortest=1:x={}:y={} [tmp{}]; '.format(i-1, i, left, top, i)
			else:
				complex_variable += '[tmp{}][s{}] overlay=shortest=1:x={}:y={} \" '.format(i-1, i, left, top)
	
		scripts += complex_variable
		scripts += '-c:v libx264 -r 10 croped_{}.avi '.format(cam_name)

		sh_file = open("scripts/ffmpeg_merge_{}.sh".format(cam_name), "w")
		sh_file.write(scripts)
		sh_file.close()


if __name__ == "__main__":

	camera_used_blocks = Optimizer.optimization_solver(cameras, cam_to_tshape, [0, 900], gt_multi_hashmap=RegressionFilter.Multi_Hashmap)

	camera_nouse_blocks = {cam: set(list(range(cam_to_tshape[cam][0] * cam_to_tshape[cam][1]))) \
							- set(camera_used_blocks[cam]) for cam in cameras}

	# Smooth camera nouse blocks
	for cam in cameras:
		block_h_n, block_w_n = cam_to_tshape[cam]
		remove_set = set()
		for block in camera_nouse_blocks[cam]:
			used_neighbors = 0
			if block % block_w_n - 1 >= 0 and block - 1 not in camera_nouse_blocks[cam]:
				used_neighbors += 1
			if block % block_w_n + 1 < block_w_n and block + 1 not in camera_nouse_blocks[cam]:
				used_neighbors += 1
			if block - block_w_n  >= 0 and block - block_w_n not in camera_nouse_blocks[cam]:
				used_neighbors += 1
			if block + block_w_n < block_h_n * block_w_n and block + block_w_n not in camera_nouse_blocks[cam]:
				used_neighbors += 1
			if used_neighbors >= 3:
				remove_set.add(block)
		print(cam, remove_set)
		if cam == 'c003':
			for block in camera_nouse_blocks[cam]:
				if block >= 70 and block <= 80:
					remove_set.add(block)
		camera_nouse_blocks[cam] = camera_nouse_blocks[cam] - remove_set
		camera_used_blocks[cam] = list(set(camera_used_blocks[cam]).union(remove_set))

	print(camera_nouse_blocks)

	camera_to_tiles = {cam: [] for cam in cameras}
	for cam in cameras:
		tile_map = [[0] * cam_to_tshape[cam][1] for _ in range(cam_to_tshape[cam][0])]
		for id in camera_used_blocks[cam]:
			tile_map[id // cam_to_tshape[cam][1]][id % cam_to_tshape[cam][1]] = 1
		result = MergeTile.mergeTiles(tile_map)
		for top, left, height, width in result:
			camera_to_tiles[cam].append(tuple([top * tile_height, left * tile_width, height * tile_height, width * tile_width]))

	for cam in camera_to_tiles:
		print(len(camera_to_tiles[cam]))
		print(camera_to_tiles[cam])

	generate_ffmpeg_crop_scripts(camera_to_tiles)
	generate_ffmpeg_merge_scripts(camera_to_tiles)

	for cam in cameras:
		Visualizer.plot_frame_w_nouse_tiles(cam, 20, camera_nouse_blocks[cam], 'images/'+ cam + 'test.jpg')
		Visualizer.plot_ROI_mask(cam, camera_nouse_blocks[cam], 'images/' + cam + '_mask.jpg')

			
				