from CreateGraph import DATA_PATH, WORKSAPCE
import math, cv2
from os import confstr_names
import numpy as np
from numpy.lib.npyio import _savez_compressed_dispatcher
import General as general

WORKSAPCE = general.WORKSAPCE
DATA_PATH = general.DATA_PATH

def generate_ffmpeg_crop_scripts(cameras_to_tiles):
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
			scripts += '-map [s{}] -c:v libx264 videos/S01/nomerge/{}/{:02d}.mp4 '.format(i, cam_name, i)
		sh_file = open("videos/S01/nomerge/{}.sh".format(cam_name), "w")
		sh_file.write(scripts)
		sh_file.close()

if __name__ == '__main__':
    source_dir = 'videos/S01/'
    target_setting = '5e-06_1.0/'
    cameras_shape = general.cameras_shape
    tile_height, tile_width = 64, 64
    cameras = general.cameras
    cameras_to_tiles = {cam_name: [] for cam_name in cameras}
    for line in open(source_dir + target_setting + 'tile.txt').readlines():
        cam_name, tile_id = line.split(' ')[0], int(line.split(' ')[1])
        f_height, f_width = cameras_shape[cam_name]
        row_n =  f_width // tile_width
        left = (tile_id % row_n) * tile_width
        top = (tile_id // row_n) * tile_height
        cameras_to_tiles[cam_name].append((top, left, 64, 64))
    generate_ffmpeg_crop_scripts(cameras_to_tiles)
