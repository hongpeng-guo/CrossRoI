import math, cv2
from os import confstr_names
import numpy as np
from numpy.lib.npyio import _savez_compressed_dispatcher
import General as general
import Optimizer
import Visualizer
import MergeTile
import SVMFilter

# environment & macro definations
FRAME_RATE = general.FRAME_RATE
WORKSAPCE = general.WORKSAPCE
DATA_PATH = general.DATA_PATH
GT_PATH = general.GT_PATH
ts_base = general.ts_base
cameras = general.cameras
cameras_shape = general.cameras_shape
tile_height, tile_width = general.tile_height, general.tile_width
cam_to_tshape = general.cam_to_tshape

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
def generate_ffmpeg_crop_scripts(cameras_to_tiles, subdir_name):
	for cam_idx, cam_name in enumerate(cameras_to_tiles):
		scripts = 'ffmpeg -i ' + WORKSAPCE + DATA_PATH + cam_name + '/' + 'vdo.avi' + ' -filter_complex '
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
			scripts += '-map [s{}] -c:v libx264 videos/{}/tmp/{}_{:02d}.mp4 '.format(i, subdir_name, cam_name, i)
		sh_file = open("scripts/crop/ffmpeg_crop_{}_{}.sh".format(cam_name, subdir_name.split('/')[1]), "w")
		sh_file.write(scripts)
		sh_file.close()


# Generate ffmpeg scripts to crop video into tiles
def generate_ffmpeg_merge_scripts(cameras_to_tiles, subdir_name):
	for cam_idx, cam_name in enumerate(cameras_to_tiles):
		scripts = 'ffmpeg '
		for i in range(len(cameras_to_tiles[cam_name])):
			scripts += '-i videos/{}/tmp/{}_{:02d}.mp4 '.format(subdir_name, cam_name, i)
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
		scripts += '-c:v libx264 -r 10 videos/{}/croped_{}.mp4 '.format(subdir_name ,cam_name)

		sh_file = open("scripts/merge/ffmpeg_merge_{}_{}.sh".format(cam_name, subdir_name.split('/')[1]), "w")
		sh_file.write(scripts)
		sh_file.close()


if __name__ == "__main__":
	experiment_setting = general.experiment_setting

	experiment_subdirs = general.experiment_subdirs

	for i, (gamma, res_thresh) in enumerate(experiment_setting):

		subdir_name = general.SCENE_NAME + '/' + experiment_subdirs[i]

		print(subdir_name, gamma, res_thresh)

		filtered_hashmap, _, _, _ = SVMFilter.get_SVM_HashMap(gamma=gamma, res_thres=res_thresh)

		camera_used_blocks = Optimizer.optimization_solver(cameras, cam_to_tshape, [0, 900], \
														gt_multi_hashmap=filtered_hashmap)

		camera_nouse_blocks = {cam: set(list(range(cam_to_tshape[cam][0] * cam_to_tshape[cam][1]))) \
								- set(camera_used_blocks[cam]) for cam in cameras}

		result_file = open('videos/' + subdir_name + '/' + 'tile.txt', 'w')
		for cam_name in camera_used_blocks:
			for block in camera_used_blocks[cam_name]:
				result_file.write('{} {}\n'.format(cam_name, block))
		result_file.close()

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

		generate_ffmpeg_crop_scripts(camera_to_tiles, subdir_name)
		generate_ffmpeg_merge_scripts(camera_to_tiles, subdir_name)

		for cam in cameras:
			Visualizer.plot_frame_w_nouse_tiles(cam, 20, camera_nouse_blocks[cam], \
												'videos/'+ subdir_name + '/' + cam + '_test.jpg')		

			Visualizer.plot_ROI_mask(cam, camera_nouse_blocks[cam], \
									 'videos/' + subdir_name + '/' + cam + '_mask.jpg')