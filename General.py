import math, cv2

# 'S01' or 'S02'
ENVIRONMENT_SCENE = 'S01' 

# environment & macro definations
FRAME_RATE = 10
WORKSAPCE = "/home/hongpeng/Desktop/research/AICity/Track3/"
DET_PATH =  "det/det_yolo3.txt"
TRACK_PATH = "mtsc/mtsc_deepsort_yolo3.txt"
GT_PATH = "gt/gt.txt"

if ENVIRONMENT_SCENE == 'S01': 
	DATA_PATH = "train/S01/" 
	TIMESTAMP_PATH = "cam_timestamp/S01.txt"
	REID_PATH = "./reid_trn.txt"
	SCENE_NAME = 'S01'
	cameras = ["c001", "c002", "c003", "c004", "c005"]
else:	
	DATA_PATH = "validation/S02/" 
	TIMESTAMP_PATH = "cam_timestamp/S02.txt"
	REID_PATH = "./reid_val.txt"
	SCENE_NAME = 'S02'
	cameras = ["c006", "c007", "c008", "c009"]

TIMESTAMP_CONTENT = open(WORKSAPCE + TIMESTAMP_PATH).read().split('\n')[:-1]
ts_base = {each.split(' ')[0]: float(each.split(' ')[1]) for each in TIMESTAMP_CONTENT}

# get the k-th frame of a certain camera
def get_frame(cam_name, frame_id):
	cap = cv2.VideoCapture( WORKSAPCE + DATA_PATH + cam_name + '/' + 'vdo.avi')
	cap.set(1, frame_id)
	_, frame = cap.read()
	return frame

cameras_shape = {camera: get_frame(camera, 0).shape[:2] for camera in cameras}
tile_height, tile_width = 64, 64
cam_to_tshape = {cam: (math.ceil(cameras_shape[cam][0] / tile_height), math.ceil(cameras_shape[cam][1] / tile_width)) for cam in cameras}

frame_diff = {cam: int(FRAME_RATE * ts_base[cam]) for cam in cameras}
frame_diff = {cam: frame_diff[cam] - max(frame_diff.values()) for cam in cameras}

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

experiment_setting = [(1e-6, 1.0), (5e-6, 1.0), (1e-5, 1.0), (5e-5, 1.0), (1e-4, 1.0), \
					  (2e-5, 0.01), (2e-5, 0.05), (2e-5, 0.1), (2e-5, 1.0),(2e-5, 10.0), \
					  (100, 200)]

experiment_subdirs = ['1e-06_1.0', '5e-06_1.0', '1e-05_1.0', '5e-05_1.0', '1e-04_1.0', \
					  '2e-05_0.01', '2e-05_0.05', '2e-05_0.1',  '2e-05_1.0', '2e-05_10.0',\
					  'nofilter' ]
