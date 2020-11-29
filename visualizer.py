import os, cv2
import numpy as np
import create_graph as cg

def plot_frame_w_nouse_tiles(cam_name, frame_id, no_use_list):
    base_frame = cg.get_frame(cam_name, frame_id)
    f_width = cg.cameras_shape[cam_name][1]
    t_height, t_width = cg.tile_height, cg.tile_width
    n_row = f_width // t_width
    for tile in no_use_list:
        left, top = (tile % n_row) * t_width, (tile // n_row) * t_height
        cv2.rectangle(base_frame, (left, top), (left + t_width, top + t_height), (255, 0, 0), -1)
    # cv2.imwrite(cam_name + '.png', base_frame)
    cv2.imshow('image', base_frame)
    cv2.waitKey()
    cv2.destroyAllWindows()

# compute the data association between two cameras
# def projection_2_cams(cam_A, cam_B, detector):
#     frame_diff = int (FRAME_RATE * (ts_base[cam_B] - ts_base[cam_A]))

#     cam_A_hashmap = data_to_hashmap(cam_A, detector)
#     cam_B_hashmap = data_to_hashmap(cam_B, detector)

#     result_match, cam_A_match, cam_B_match = [], [], []
#     for key in cam_A_hashmap:
#         if key[0] > 600:
#             break
#         if (key[0] - frame_diff, key[1]) in cam_B_hashmap:
#             result_match.append((cam_A_hashmap[key], cam_B_hashmap[(key[0] - frame_diff, key[1])]))
#             cam_A_match.append(key)
#             cam_B_match.append((key[0] - frame_diff, key[1]))
#     cam_A_unique = [cam_A_hashmap[key] for key in cam_A_hashmap if key not in cam_A_match and key[0] < 600]
#     cam_B_unique = [cam_B_hashmap[key] for key in cam_B_hashmap if key not in cam_B_match and key[0] + frame_diff < 600]
#     print(len(result_match), len(cam_A_unique), len(cam_B_unique))
#     return result_match, cam_A_unique, cam_B_unique

# for base_frame in range(100, 101):
#     c001_to_c002, c001_unique, c002_unique = projection_2_cams('c001', 'c002', 'rcnn')
#     cap_c001 = cv2.VideoCapture( WORKSAPCE + DATA_PATH + 'c001' + '/' + 'vdo.avi' )
#     cap_c001.set(1, base_frame)
#     _, c001_frame = cap_c001.read()
#     cap_c002 = cv2.VideoCapture( WORKSAPCE + DATA_PATH + 'c004' + '/' + 'vdo.avi' )
#     cap_c002.set(1, base_frame - round (FRAME_RATE * (ts_base['c004'] - ts_base['c001'])))
#     _, c002_frame = cap_c002.read()

#     for rec1 in c001_unique:
#         left1, top1, width1, height1, confidence1 = rec1[0], rec1[1], rec1[2], rec1[3], rec1[4]
#         cv2.rectangle(c001_frame, (left1, top1), (left1 + width1, top1 + height1), (255, 0, 0), 1)
#         cv2.circle(c001_frame, (left1 + width1//2, top1 + height1//2), 2, (0, 0, 255), -1)
#     cv2.imshow('image', c001_frame)
#     cv2.waitKey()
#     cv2.destroyAllWindows()

#     for rec1 in c002_unique:
#         left1, top1, width1, height1, confidence1 = rec1[0], rec1[1], rec1[2], rec1[3], rec1[4]
#         cv2.rectangle(c002_frame, (left1, top1), (left1 + width1, top1 + height1), (255, 0, 0), 1)
#         cv2.circle(c002_frame, (left1 + width1//2, top1 + height1//2), 2, (0, 0, 255), -1)
#     cv2.imshow('image', c002_frame)
#     cv2.waitKey()
#     cv2.destroyAllWindows()

#     # img = cv2.hconcat([c001_frame,c002_frame])
#     # height, width = img.shape[:2]
#     # thumbnail = cv2.resize(img, (round(width / 2), round(height / 2)), interpolation=cv2.INTER_AREA)

#     # height, width = c001_frame.shape[:2]
#     # blank1 = np.zeros((height,width,3), np.uint8)
#     # blank2 = np.zeros((height,width,3), np.uint8)

#     # for rec1, rec2 in c001_to_c002:
#     #     left1, top1, width1, height1, confidence1 = rec1[0], rec1[1], rec1[2], rec1[3], rec1[4]
#     #     left2, top2, width2, height2, confidence2 = rec2[0], rec2[1], rec2[2], rec2[3], rec2[4]
#     #     blank1[top1: top1+height1, left1: left1+width1] = [50, 50, 50]
#     #     blank2[top2: top2+height2, left2: left2+width2] = [50, 50, 50]

#     # for rec1, rec2 in c001_to_c002:
#     #     left1, top1, width1, height1, confidence1 = rec1[0], rec1[1], rec1[2], rec1[3], rec1[4]
#     #     left2, top2, width2, height2, confidence2 = rec2[0], rec2[1], rec2[2], rec2[3], rec2[4]
#     #     if confidence1 >= confidence2:
#     #         blank1[top1: top1+height1, left1: left1+width1, 0] = 100
#     #         blank2[top2: top2+height2, left2: left2+width2, 0] = 100
#     #     else: 
#     #         blank1[top1: top1+height1, left1: left1+width1, 1] = 100
#     #         blank2[top2: top2+height2, left2: left2+width2, 1] = 100

#     # filter_img = cv2.hconcat([blank1, blank2])
#     # height, width = filter_img.shape[:2]
#     # filter_thumbnail = cv2.resize(filter_img, (round(width / 2), round(height / 2)), interpolation=cv2.INTER_AREA)
#     # img = cv2.hconcat([c001_frame,c002_frame])
#     # thumbnail = cv2.resize(img, (round(width / 2), round(height / 2)), interpolation=cv2.INTER_AREA)

#     # thumbnail = cv2.addWeighted(thumbnail, 0.25, filter_thumbnail, 1.0, 1)

#     # cv2.imshow('image', thumbnail)
#     # cv2.waitKey()
#     # cv2.destroyAllWindows()


# base_frame = 100
# _, unique_1_2, _ = projection_2_cams('c002', 'c001', 'rcnn')
# _, unique_1_3, _ = projection_2_cams('c002', 'c003', 'rcnn')
# _, unique_1_4, _ = projection_2_cams('c002', 'c004', 'rcnn')
# _, unique_1_5, _ = projection_2_cams('c002', 'c005', 'rcnn')
# cap_c001 = cv2.VideoCapture( WORKSAPCE + DATA_PATH + 'c002' + '/' + 'vdo.avi' )
# cap_c001.set(1, base_frame)
# _, c001_frame = cap_c001.read()

# unique = list(set(unique_1_2).intersection(set(unique_1_3), set(unique_1_4), set(unique_1_5)))
# print(len(unique_1_2), len(unique_1_3), len(unique_1_4), len(unique_1_5), len(unique))

# for rec1 in unique:
#     left1, top1, width1, height1, confidence1 = rec1[0], rec1[1], rec1[2], rec1[3], rec1[4]
#     cv2.rectangle(c001_frame, (left1, top1), (left1 + width1, top1 + height1), (255, 0, 0), 1)
#     cv2.circle(c001_frame, (left1 + width1//2, top1 + height1//2), 2, (0, 0, 255), 1)
# cv2.imshow('image', c001_frame)
# cv2.waitKey()
# cv2.destroyAllWindows()