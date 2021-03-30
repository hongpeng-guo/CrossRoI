import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math

from numpy.testing._private.utils import clear_and_catch_warnings
import General as general
import bitrate_stats as bt

# BitrateStats

reducto_acc = general.reducto_acc
reducto_names = ['0.8', '0.9', '0.95', '1.0']
Results = {acc: {} for acc in reducto_acc}
cameras = general.cameras

roi_setting = '2e-05_0.01'

roi_network_res = np.load('results/S01/network/results.npy', allow_pickle=True).item()[roi_setting]
roi_server_res = np.load('results/S01/server_speed/speed_5.npy', allow_pickle=True).item()[roi_setting]
roi_camera_res = np.load('results/S01/camera_speed/result.npy', allow_pickle=True).item()[roi_setting]
roi_acc = np.load('results/S01/accuracy/roi/' + roi_setting + '.npy')[300:]
baseline_acc = np.load('results/S01/accuracy/baseline.npy')[300:]
baseline_network_res = np.load('results/S01/network/results.npy', allow_pickle=True).item()['baseline']
baseline_server_res = np.load('results/S01/server_speed/speed_5.npy', allow_pickle=True).item()['baseline']
baseline_camera_res = np.load('results/S01/camera_speed/result.npy', allow_pickle=True).item()['baseline']

for i, setting_name in enumerate(reducto_names):
    Results[reducto_acc[i]] = np.load('results/S01/accuracy/reducto/' + setting_name + '.npy', allow_pickle=True).item()

network_res = np.load('results/S01/network/results.npy', allow_pickle=True).item()
seg_amplyfy = {cam: {ck: 1 for ck in [1, 1.5, 2, 2.5, 3, 10]} for cam in cameras}
for cam in cameras:
    for ck in [1, 1.5, 2, 2.5, 3, 10]:
        seg_amplyfy[cam][ck] = network_res['segments'][(cam, ck)]['avg_bitrate'] / \
                                network_res['segments'][(cam, 10)]['avg_bitrate']

full_video_frames, roi_video_frames = {cam: [] for cam in cameras}, {cam: [] for cam in cameras}
for cam in cameras:
    full_br = bt.BitrateStats('videos/S01/baseline/'+'h264_'+cam+'.mp4', chunk_size=1)
    full_br.calculate_statistics()
    full_video_frames[cam] = [frame['size'] * seg_amplyfy[cam][1] for frame in full_br.frames]

    roi_br = bt.BitrateStats('videos/S01/2e-05_0.01/'+'croped_'+cam+'.mp4', chunk_size=1)
    roi_br.calculate_statistics()
    roi_video_frames[cam] = [frame['size'] * seg_amplyfy[cam][1] for frame in roi_br.frames]


Plot = {acc:{'reducto':{}, 'reductoroi':{}} for acc in reducto_acc}
for acc in reducto_acc:
    for method in ['reducto', 'reductoroi']:
        # Accuracy
        Plot[acc][method]['accuracy'] = 1 - np.mean(np.abs(np.array(Results[acc][method]['count'][300:]) - np.array(baseline_acc)) / np.array(baseline_acc))
        print(str(acc), method, Plot[acc][method]['accuracy'])
        # Frame count
        Plot[acc][method]['frame_count'] = np.sum([len(Results[acc][method]['frames'][key]) for key in Results[acc][method]['frames']])
        print(6000 - Plot[acc][method]['frame_count'])
        # Network
        frame_data = full_video_frames if method == 'reducto' else roi_video_frames
        total_net = []
        for cam in cameras:
            selected_frames = Results[acc][method]['frames'][cam]
            f_sizes = [frame_data[cam][i] if i in selected_frames else 0 for i in range(601, 1801)]
            f_sizes = np.array(f_sizes).reshape(10, 120)
            for i in range(120):
                if np.count_nonzero(f_sizes[:, i]) < 5:
                    f_sizes[:,i] *= seg_amplyfy[cam][1]
            f_sizes = np.mean(f_sizes, axis=1)
            total_net.append(f_sizes)
        total_net = np.vstack(total_net).sum(axis=0)
        Plot[acc][method]['network'] = (np.mean(total_net)/10000, np.std(total_net)/10000)
        print(Plot[acc][method]['network'])

        # Sever Throughput
        server_data = baseline_server_res if method == 'reducto'else roi_server_res
        server_data = np.mean(np.vstack([server_data[key] for key in server_data]), axis=0)
        new_server_data =  6000 / Plot[acc][method]['frame_count'] * np.mean(server_data) * 2
        Plot[acc][method]['server'] = (np.mean(new_server_data), np.std(new_server_data)) 
        print(Plot[acc][method]['server'])

        # Camera Throughput
        camera_data = baseline_camera_res if method == 'reducto'else roi_camera_res
        camera_data = np.mean([camera_data[key] for key in camera_data], axis=0)
        new_camera_data =  6000 / Plot[acc][method]['frame_count'] * np.mean(camera_data) * 10
        Plot[acc][method]['camera'] = (np.mean(new_camera_data), np.std(new_camera_data)) 
        print(Plot[acc][method]['camera'])

        # End to End Delay
        Plot[acc][method]['delay'] = 0.5 + 5 / Plot[acc][method]['camera'][0] + \
                                     0.005 + Plot[acc][method]['network'][0] / 30 + \
                                     25 / Plot[acc][method]['server'][0]
        print(Plot[acc][method]['delay'])
