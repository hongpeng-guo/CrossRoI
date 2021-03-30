import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
import General as general

params = {
   'axes.labelsize': 'xx-large',
   'legend.fontsize': 'xx-large',
   'axes.titlesize': 'xx-large',
   'xtick.labelsize': 'xx-large',
   'ytick.labelsize': 'xx-large',
   'text.usetex': False,
   "savefig.dpi": 600
}

matplotlib.rcParams.update(params)

org_window = [300, 1800]

new_color = {'acc': '#fbb4ae',
             'network': '#b3cde3',
             'CDelay': '#2c7bb6',
             'NDelay': '#ffffbf',
             'SDelay': '#abd9e9'}

def autolabel(rects, ax, bit=3, pos='center'):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        if height % 1 == 0:
            ax.annotate(height,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),  # 3 points vertical offset
                        textcoords="offset points", size='xx-large',
                        ha=pos, va='bottom')
        elif bit == 3:
            ax.annotate('{:.3f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),  # 3 points vertical offset
                        textcoords="offset points", size='xx-large',
                        ha=pos, va='bottom')
        else:
            ax.annotate('{:.1f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),  # 3 points vertical offset
                        textcoords="offset points", size='xx-large',
                        ha=pos, va='bottom')



def plot_bar_acc(acc_arr, x_ticks, x_label,  filename, N=5, x_ticklabels=None):
     
    width = 0.4
    fig, ax = plt.subplots()
    x = np.arange(len(x_ticks))
    ax.set_xticks(x)
    if x_ticklabels == None:
        ax.set_xticklabels(x_ticks)
    else:
        ax.set_xticklabels(x_ticklabels)

    rects = []

    for i in range(N):
        rect = ax.bar( i, acc_arr[i][0], width=width, edgecolor='black', color=new_color['acc'])
        rects.append(rect)
    
    ax.set_ylim(0.98, 1.002)
    ax.set_yticks((.98, .99, 1.0))
    ax.yaxis.grid(linestyle='--', which='both')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel(x_label)

    for rect in rects:
        autolabel(rect, ax)

    ratio = 0.8
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    ax.set_aspect(1.0/ax.get_data_ratio()*ratio)

    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig('figures/'+ filename, pad_inches = 0, bbox_inche='tight')
    plt.show()


def plot_bar_network(network_arr, x_ticks, x_label, filename, N=5, x_ticklabels=None):
     
    width = 0.4
    fig, ax = plt.subplots()
    x = np.arange(len(x_ticks))
    ax.set_xticks(x)
    if x_ticklabels == None:
        ax.set_xticklabels(x_ticks)
    else:
        ax.set_xticklabels(x_ticklabels)

    rects = []
    for i in range(N):
        rect = ax.bar( i, network_arr[i][0], yerr=0.5 * network_arr[i][1], width=width, edgecolor='black', color=new_color['network'])
        rects.append(rect)

    ax.yaxis.grid(linestyle='--', which='both')
    ax.set_ylabel('Network Overhead (Mbps)')
    ax.set_xlabel(x_label)

    ratio = 0.8
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    ax.set_aspect(1.0/ax.get_data_ratio()*ratio)


    for rect in rects:
        autolabel(rect, ax, bit=1)

    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig('figures/'+ filename, pad_inches = 0, bbox_inche='tight')
    plt.show()


def plot_bar_delay(Cdelay_arr, Ndelay_arr, Sdelay_arr, x_ticks, x_label, filename, N=5, x_ticklabels=None):
     
    width = 0.25
    fig, ax = plt.subplots()
    x = np.arange(len(x_ticks))
    ax.set_xticks(x)
    if x_ticklabels == None:
        ax.set_xticklabels(x_ticks)
    else:
        ax.set_xticklabels(x_ticklabels)

    height_max = 0

    rects = []
    for i in range(N):
        Slabel = 'Server Delay' if i==0 else None
        rect1 = ax.bar( i, Cdelay_arr[i][0] + Ndelay_arr[i][0] + Sdelay_arr[i][0], \
                yerr=Cdelay_arr[i][1] + Ndelay_arr[i][1] + Sdelay_arr[i][1],\
                width=width, edgecolor='black', color=new_color['SDelay'], label=Slabel)
        rects.append(rect1)
        height_max = max(height_max, Cdelay_arr[i][0] + Ndelay_arr[i][0] + Sdelay_arr[i][0])

        Nlabel = 'Network Delay' if i==0 else None
        rect2 = ax.bar( i, Cdelay_arr[i][0] + Ndelay_arr[i][0] , hatch='*',\
                width=width, edgecolor='black', color=new_color['NDelay'], label=Nlabel)

        Clabel = 'Camera Delay' if i==0 else None
        rect3 = ax.bar( i, Cdelay_arr[i][0], hatch='\\',\
                width=width, edgecolor='black', color=new_color['CDelay'], label=Clabel)
    
    ax.set_ylim(0, height_max + 0.5)
    ax.yaxis.grid(linestyle='--', which='both')
    ax.set_ylabel('End to End Delay (s)')
    ax.set_xlabel(x_label)

    ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2)

    for rect in rects:
        autolabel(rect, ax)

    ratio = 0.6
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    ax.set_aspect(1.0/ax.get_data_ratio()*ratio)
    plt.savefig('figures/'+ filename, pad_inches = 0, bbox_inche='tight')
    plt.show()

  


if __name__ == '__main__':

    cameras = general.cameras

    accuracy_dir = 'results/S01/accuracy/'
    network_res = np.load('results/S01/network/results.npy', allow_pickle=True).item()
    server_res = np.load('results/S01/server_speed/speed_5.npy', allow_pickle=True).item()
    camera_res = np.load('results/S01/camera_speed/result.npy', allow_pickle=True).item()


    setting_names = general.experiment_subdirs[:-1]
    setting_params = general.experiment_setting[:-1]


    network_res[setting_names[7]], network_res[setting_names[8]] = network_res[setting_names[8]], network_res[setting_names[7]]
    server_res[setting_names[7]], server_res[setting_names[8]] = server_res[setting_names[8]], server_res[setting_names[7]]
    camera_res[setting_names[7]], camera_res[setting_names[8]] = camera_res[setting_names[8]], camera_res[setting_names[7]]


    baseline_acc = np.load(accuracy_dir + 'baseline.npy')[:1200]

    seg_amplyfy = {cam: {ck: 1 for ck in [1, 1.5, 2, 2.5, 3, 10]} for cam in cameras}
    for cam in cameras:
        for ck in [1, 1.5, 2, 2.5, 3, 10]:
            seg_amplyfy[cam][ck] = network_res['segments'][(cam, ck)]['avg_bitrate'] / \
                                   network_res['segments'][(cam, 10)]['avg_bitrate']

    acc_data, network_data, server_data, camera_data = [], [], [], []
    
    for setting in setting_names:
        acc_data.append(np.load(accuracy_dir + 'roi/' + setting + '.npy')[:1200] / baseline_acc)

        queue = []
        for cam in cameras:
            queue.append( np.array(network_res[setting][(cam, 1)]['bitrate_per_chunk'][30:150]) * seg_amplyfy[cam][1] )
        queue = np.array(queue)
        queue = queue.sum(axis=0) / 1000
        network_data.append(queue)  

        queue = []
        for cam in cameras:
            queue.append( np.array(np.array(server_res[setting][cam])) * 2)
        queue = np.array(queue)
        queue = queue.mean(axis=0)
        server_data.append(queue)

        queue = []
        for cam in cameras:
            queue.append( np.array(np.array(camera_res[setting][cam])) * 10)
        queue = np.array(queue)
        queue = queue.mean(axis=0)
        camera_data.append(queue)  

    camera_delay = 5 / np.array(camera_data) + 0.5
    network_delay = 0.05 + np.array(network_data) / 30
    server_delay = 25 / np.array(server_data)

    acc_plot = [(np.mean(acc_data[i]), np.std(acc_data[i])) for i in range(len(acc_data))]
    network_plot = [(np.mean(network_data[i]), np.std(network_data[i])) for i in range(len(network_data))]
    server_plot = [(np.mean(server_data[i]), np.std(server_data[i])) for i in range(len(server_data))]
    camera_plot = [(np.mean(camera_data[i]), np.std(camera_data[i])) for i in range(len(camera_data))]
    camera_delay_plot = [(np.mean(camera_delay[i]), np.std(camera_delay[i])) for i in range(len(camera_delay))]
    network_delay_plot = [(np.mean(network_delay[i]), np.std(network_delay[i])) for i in range(len(network_delay))]
    server_delay_plot = [(np.mean(server_delay[i]), np.std(server_delay[i])) for i in range(len(server_delay))]


    # segment parameter experiment
    seg_setting= '2e-05_0.01'
    seg_setting_id = 5
    seg_network_data = []
    for segment_length in [1, 1.5, 2, 2.5, 3, 10]:
        queue = []
        for cam in cameras:
            queue.append( np.array(network_res[seg_setting][(cam, segment_length)]['bitrate_per_chunk']) * seg_amplyfy[cam][segment_length] )
        min_length = min([len(queue[i]) for i in range(len(queue))])
        for i in range(len(queue)):
            queue[i] = queue[i][:min_length]
        queue = np.array(queue)
        queue = queue.sum(axis=0) / 1000
        seg_network_data.append(queue)
    seg_network_plot = [(np.mean(seg_network_data[i]), np.std(seg_network_data[i])) for i in range(len(seg_network_data))]
    
    seg_camera_delay, seg_network_delay, seg_server_delay = [], [], []
    for i, segment_length in enumerate([1, 1.5, 2, 2.5, 3, 10]):
        seg_camera_delay.append(0.5 * (segment_length + 10 * segment_length / camera_data[seg_setting_id]))
        seg_network_delay.append(0.05 + segment_length * seg_network_data[i] / 30)
        seg_server_delay.append(0.5 * segment_length * 50 / server_data[i])

    seg_camera_delay_plot = [(np.mean(seg_camera_delay[i]), np.std(seg_camera_delay[i])) for i in range(len(seg_camera_delay))]
    seg_network_delay_plot = [(np.mean(seg_network_delay[i]), np.std(seg_network_delay[i])) for i in range(len(seg_network_delay))]
    seg_server_delay_plot = [(np.mean(seg_server_delay[i]), np.std(seg_server_delay[i])) for i in range(len(seg_server_delay))]

    print(seg_network_plot)
    print(seg_camera_delay_plot)
    print(seg_network_delay_plot)
    print(seg_server_delay_plot)

    plot_bar_acc(acc_plot[:5], [1e-6, 5e-6, 1e-5, 5e-5, 1e-4], r'$\gamma$', 'svm-acc.eps',\
                 x_ticklabels=['1e-6', '5e-6', '1e-5', '5e-5', '1e-4'])
    plot_bar_acc(acc_plot[5:10], [0.01, 0.05, 0.1, 1.0, 10.0], r'$\theta$', 'regression-acc.eps')

    plot_bar_network(network_plot[:5], [1e-6, 5e-6, 1e-5, 5e-5, 1e-4], r'$\gamma$', 'svm-network.eps',\
                 x_ticklabels=['1e-6', '5e-6', '1e-5', '5e-5', '1e-4'])
    plot_bar_network(network_plot[5:10], [0.01, 0.05, 0.1, 1.0, 10.0], r'$\theta$', 'regression-network.eps')
    plot_bar_network(seg_network_plot[:5], [1, 1.5, 2, 2.5, 3], 'Segment Length (s)', 'segment-network.eps')

    plot_bar_delay(camera_delay_plot[:5], network_delay_plot[:5], server_delay_plot[:5], \
                    [1e-6, 5e-6, 1e-5, 5e-5, 1e-4], r'$\gamma$',  'svm-delay.eps',\
                 x_ticklabels=['1e-6', '5e-6', '1e-5', '5e-5', '1e-4'])
    plot_bar_delay(camera_delay_plot[5:10], network_delay_plot[5:10], server_delay_plot[5:10], \
                    [0.01, 0.05, 0.1, 1.0, 10.0], r'$\theta$',  'regression-delay.eps')
    plot_bar_delay(seg_camera_delay_plot[:5], seg_network_delay_plot[:5], seg_server_delay_plot[:5], \
                    [1, 1.5, 2, 2.5, 3], 'Segment Length (s)',  'segment-delay.eps')