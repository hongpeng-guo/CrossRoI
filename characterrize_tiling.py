from CreateGraph import DATA_PATH, WORKSAPCE
import math, cv2
from os import confstr_names
import numpy as np
from numpy.lib.npyio import _savez_compressed_dispatcher
import General as general

WORKSAPCE = general.WORKSAPCE
DATA_PATH = general.DATA_PATH
cameras_shape = general.cameras_shape

def generate_ffmpeg_crop_scripts(cam_name, h, w):
    width = cameras_shape[cam_name][1]//w
    height = cameras_shape[cam_name][0]//h

    scripts = 'ffmpeg -i ' + WORKSAPCE + DATA_PATH + cam_name + '/' + 'vdo.avi' + ' -filter_complex '
    complex_variable = '\"[0]split={}'.format(h * w)
    for i in range(h * w):
        complex_variable += '[s{}]'.format(i)
    complex_variable += '; '
    for i in range(h * w):
        left, top = (i % w) * width, (i // w) * height 
        complex_variable += '[s{}]crop={}:{}:{}:{}[s{}]; '.format(i, width, height, left, top, i)
    complex_variable = complex_variable[:-2] + '\" '
    scripts += complex_variable
    for i in range(h * w):
        scripts += '-map [s{}] -c:v libx264 {}/{}_{}/{:02d}.mp4 '.format(i, cam_name, h, w, i)
    print(scripts)
    sh_file = open("../../video_profile/{}_{}_{}.sh".format(cam_name, h, w), "w")
    sh_file.write(scripts)
    sh_file.close()

if __name__ == '__main__':
    for camera in general.cameras:
        for h, w in [(2, 2), (2, 4), (4, 4), (4, 8), (8, 8)]:
            generate_ffmpeg_crop_scripts(camera, h, w)
