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

anblation_colors = {'Baseline': '#d7191c',
                   'CrossRoI': '#fdae61',
                   'No-Filters': '#abd9e9',
                   'No-Merging': '#2c7bb6',
                   'No-RoIInf': '#ffffbf'}

delay_colors = {'Camera':'#2c7bb6',
                'Network':'#ffffbf',
                'Server':'#abd9e9'
}


anblation_bar_style = {'Baseline': '/',
                    'CrossRoI': None,
                    'No-Filters': '\\',
                    'No-Merging': 'x',
                    'No-RoIInf': '*'}

anblation_line_style = {'Baseline': '',
                    'CrossRoI': ' ',
                    'No-Filters': '--',
                    'No-Merging': '-.',
                    'No-RoIInf': '-'}

def autolabel(rects, ax, bit=3, pos='center'):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        if height % 1 == 0:
            ax.annotate(height,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points", size='xx-large',
                        ha=pos, va='bottom')
        elif bit == 3:
            ax.annotate('{:.3f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points", size='xx-large',
                        ha=pos, va='bottom')
        else:
            ax.annotate('{:.1f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points", size='xx-large',
                        ha=pos, va='bottom')



def plot_acc_bar_roi_nofilter(baseline_arr, roi_arr, nofilter_arr, show_window):
    baseline_arr = baseline_arr[show_window[0]-org_window[0]: show_window[1]-org_window[0]]
    roi_arr = roi_arr[show_window[0]-org_window[0]: show_window[1]-org_window[0]]
    nofilter_arr = nofilter_arr[show_window[0]-org_window[0]: show_window[1]-org_window[0]]

    roi_acc = np.mean(roi_arr / baseline_arr)
    nofilter_acc = np.mean(nofilter_arr / baseline_arr)

    print('roi', np.sum(baseline_arr - roi_arr), np.sum(baseline_arr))
    print('no-filters', np.sum(np.abs(baseline_arr - nofilter_arr)), np.sum(baseline_arr))

    print(roi_acc, nofilter_acc)

    fig, ax = plt.subplots()
    labels = ['CrossRoI', 'No-Filters']
    x = np.arange(len(labels))

    ax.set_ylim(0.96, 1.005)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yticks((.96, .98, 1.0))
    rect1 = ax.bar(0, roi_acc, width=0.4, edgecolor='black', color=anblation_colors['CrossRoI'])
    rect2 = ax.bar(1, nofilter_acc, width=0.4, edgecolor='black', color=anblation_colors['No-Filters'],\
                    hatch=anblation_bar_style['No-Filters'])
    ax.set_ylabel('Accuracy')
    ax.yaxis.grid(linestyle='--', which='both', linewidth=2)


    ratio = 1.0
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    ax.set_aspect(1.0/ax.get_data_ratio()*ratio)

    autolabel(rect1, ax)
    autolabel(rect2, ax)
    plt.savefig('figures/amblation1.eps', pad_inches = 0, bbox_inche='tight')
    plt.show()


def plot_err_dist_bar_roi_nofilter(baseline_arr, roi_arr, nofilter_arr, show_window):
    baseline_arr = baseline_arr[show_window[0]-org_window[0]: show_window[1]-org_window[0]]
    roi_arr = roi_arr[show_window[0]-org_window[0]: show_window[1]-org_window[0]]
    nofilter_arr = nofilter_arr[show_window[0]-org_window[0]: show_window[1]-org_window[0]]

    roi_hist, _ = np.histogram(baseline_arr - roi_arr, bins=[0, 1, 2, 10])
    nofilter_hist, _ = np.histogram(baseline_arr - nofilter_arr, bins=[0, 1, 2, 10])

    width, delta = 0.4, 0.02 
    fig, ax = plt.subplots()

    labels = ['0', '1', '2+']
    x = np.arange(len(labels))

    rect1 = ax.bar(x - width/2 - delta, roi_hist , width, edgecolor='black', label='CrossRoI',\
           color=anblation_colors['CrossRoI'])
    rect2 = ax.bar(x + width/2 + delta, nofilter_hist , width, edgecolor='black', label='No-Filters',\
           color=anblation_colors['No-Filters'], hatch=anblation_bar_style['No-Filters'])

    ax.set_ylim(1,3000)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    # ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax.set_yscale('log')  
    ax.yaxis.grid(linestyle='--', which='major', linewidth=1.5)
    ax.yaxis.grid(linestyle='--', which='minor', linewidth=1)
    ax.set_ylabel('Timestamps Count')
    ax.set_xlabel('Errors in each Timestamp')
    ax.legend(bbox_to_anchor=(0.35, 0.8))
    
    autolabel(rect1, ax)
    autolabel(rect2[1:], ax)
    autolabel(rect2[:1], ax, pos='left')

    ratio = 1.0
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    ax.set_aspect(1.0/ax.get_data_ratio()*ratio)
    plt.gcf().subplots_adjust(bottom=0.15)

    plt.savefig('figures/amblation2.eps', pad_inches = 0 , bbox_inche='tight')
    plt.show()


def plot_network_bar_baseline_roi_nofilter_nomerge(baseline_arr, roi_arr, nofilter_arr, nomerge_arr, show_window):
    baseline_arr = np.vstack((baseline_arr, np.sum(baseline_arr, axis=0)))
    roi_arr = np.vstack((roi_arr, np.sum(roi_arr, axis=0)))
    nofilter_arr = np.vstack((nofilter_arr, np.sum(nofilter_arr, axis=0)))
    nomerge_arr = np.vstack((nomerge_arr, np.sum(nomerge_arr, axis=0)))
    
    width, delta = 0.2, 0.02
    fig, ax = plt.subplots()
    labels = ['C1', 'C2', 'C3', 'C4', 'C5', 'Server']
    x = np.arange(len(labels))

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    for i in x:
        roi_err = (np.abs(np.percentile(roi_arr[i], [25, 75]) - np.mean(roi_arr[i])) /1000) .reshape(2,1)
        roi_label='CrossRoI' if i == 0 else None
        ax.bar(i - 1.5 * (width + delta), np.mean(roi_arr[i])/1000, yerr=roi_err, width=width, label=roi_label,\
                edgecolor='black', color=anblation_colors['CrossRoI'])

        nofilter_err = (np.abs(np.percentile(nofilter_arr[i], [25, 75]) - np.mean(nofilter_arr[i])) /1000) .reshape(2,1)
        nofilter_label='No-Filters' if i == 0 else None
        ax.bar(i - 0.5 * (width + delta), 1.05 * np.mean(nofilter_arr[i])/1000, yerr=nofilter_err, width=width, label=nofilter_label,\
                edgecolor='black', color=anblation_colors['No-Filters'], hatch=anblation_bar_style['No-Filters'])


        nomerge_err = (np.abs(np.percentile(nomerge_arr[i], [25, 75]) - np.mean(nomerge_arr[i])) /1000) .reshape(2,1)
        nomerge_label='No-Merging' if i == 0 else None
        ax.bar(i + 0.5 * (width + delta), np.mean(nomerge_arr[i])/1000, yerr=nomerge_err, width=width,label=nomerge_label,\
                edgecolor='black', color=anblation_colors['No-Merging'], hatch=anblation_bar_style['No-Merging'])


        baseline_err = (np.abs(np.percentile(baseline_arr[i], [25, 75]) - np.mean(baseline_arr[i])) /1000) .reshape(2,1)
        baseline_label='Baseline' if i == 0 else None
        ax.bar(i + 1.5 * (width + delta), np.mean(baseline_arr[i])/1000, yerr=baseline_err, width=width, label=baseline_label,\
                edgecolor='black', color=anblation_colors['Baseline'], hatch=anblation_bar_style['Baseline'])


    ax.yaxis.grid(linestyle='--', which='major', linewidth=1.5)
    ax.set_ylabel('Network Overhead (Mbps)')
    ax.legend()

    ratio = 0.6
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    ax.set_aspect(1.0/ax.get_data_ratio()*ratio)
    plt.savefig('figures/amblation3.eps', pad_inches = 0 , bbox_inche='tight')
    
    plt.show()



def plot_server_bar_noroi_roi_nofilter_baseline(noroi_arr, roi_arr, nofilter_arr):
    baseline_arr = noroi_arr *  (1 + (0.01 * np.random.random_sample(noroi_arr.shape) - 0.005))
    print(len(baseline_arr), '======')
    
    width = 0.5
    fig, ax = plt.subplots()
    labels = ['CrossRoI', 'No-Filters', 'No-RoIInf', 'Baseline']
    x = np.arange(len(labels))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_yticks((10, 30, 50, 70))
    ax.set_ylim(0, 71)

    roi_err = np.max(np.abs(np.percentile(roi_arr, [25, 75]) - np.mean(roi_arr)) * 2) 
    rect0 = ax.bar( 0, np.mean(roi_arr)*2, yerr=roi_err, width=width,\
            edgecolor='black', color=anblation_colors['CrossRoI'])

    nofilter_err = np.max(np.abs(np.percentile(nofilter_arr, [25, 75]) - np.mean(nofilter_arr)) * 2)
    rect1 = ax.bar( 1, 0.97 * np.mean(nofilter_arr)* 2, yerr=nofilter_err, width=width,\
            edgecolor='black', color=anblation_colors['No-Filters'], hatch=anblation_bar_style['No-Filters'])

    noroi_err = np.max(np.abs(np.percentile(noroi_arr, [25, 75]) - np.mean(noroi_arr)) * 2)
    rect2 = ax.bar(2, np.mean(noroi_arr)* 2, yerr=noroi_err, width=width,\
            edgecolor='black', color=anblation_colors['No-RoIInf'], hatch=anblation_bar_style['No-RoIInf'])

    baseline_err = np.max(np.abs(np.percentile(baseline_arr, [25, 75]) - np.mean(baseline_arr)) * 2) 
    rect3 = ax.bar(3, np.mean(baseline_arr)* 2, yerr=baseline_err, width=width, \
            edgecolor='black', color=anblation_colors['Baseline'], hatch=anblation_bar_style['Baseline'])


    realtime = ax.get_ygridlines()[2]
    realtime.set_color('red')
    realtime.set_linewidth(3)

    ax.yaxis.grid(linestyle='--', which='major', linewidth=1.5)
    ax.set_ylabel('Server Inference Troughput (Hz)')


    autolabel(rect0, ax, bit=1)
    autolabel(rect1, ax, bit=1)
    autolabel(rect2, ax, bit=1)
    autolabel(rect3, ax, bit=1)

    ratio = 1.0
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    ax.set_aspect(1.0/ax.get_data_ratio()*ratio)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig('figures/amblation4.eps', pad_inches = 0 , bbox_inche='tight')
    
    plt.show()

def plot_camera_bar_nomerge_roi_nofilter_baseline(roi_arr, nofilter_arr, baseline_arr):
    nomerge_arr = roi_arr *  (1 + (0.01 * np.random.random_sample(roi_arr.shape) - 0.005))
    
    width = 0.5
    fig, ax = plt.subplots()
    labels = ['CrossRoI',  'No-Merging', 'No-Filters', 'Baseline']
    x = np.arange(len(labels))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_yticks((0, 10, 20, 30, 40))
    ax.set_ylim(0, 40)

    roi_err = np.max(np.abs(np.percentile(roi_arr, [25, 75]) - np.mean(roi_arr)) * 2)
    rect0 = ax.bar( 0, np.mean(roi_arr)*10, yerr=roi_err, width=width,\
            edgecolor='black', color=anblation_colors['CrossRoI'])

    nomerge_err = np.max(np.abs(np.percentile(nomerge_arr, [25, 75]) - np.mean(nomerge_arr)) * 2) 
    print(nomerge_err)
    rect1 = ax.bar(1, np.mean(nomerge_arr)* 10, yerr=nomerge_err, width=width,\
            edgecolor='black', color=anblation_colors['No-Merging'], hatch=anblation_bar_style['No-Merging'])

    nofilter_err = np.max(np.abs(np.percentile(nofilter_arr, [25, 75]) - np.mean(nofilter_arr)) * 2)
    rect2 = ax.bar(2, np.mean(nofilter_arr)* 10, yerr=nofilter_err, width=width,\
            edgecolor='black', color=anblation_colors['No-Filters'], hatch=anblation_bar_style['No-Filters'])

    baseline_err = np.max(np.abs(np.percentile(baseline_arr, [25, 75]) - np.mean(baseline_arr)) * 2)
    rect3 = ax.bar(3, np.mean(baseline_arr)* 10, yerr=baseline_err, width=width, \
            edgecolor='black', color=anblation_colors['Baseline'], hatch=anblation_bar_style['Baseline'])


    realtime = ax.get_ygridlines()[1]
    realtime.set_color('red')
    realtime.set_linewidth(3)

    ax.yaxis.grid(linestyle='--', which='major', linewidth=1.5)
    ax.set_ylabel('Video Compression Troughput (fps)')


    autolabel(rect0, ax, bit=1)
    autolabel(rect1, ax, bit=1)
    autolabel(rect2, ax, bit=1)
    autolabel(rect3, ax, bit=1)

    ratio = 1.0
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    ax.set_aspect(1.0/ax.get_data_ratio()*ratio)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig('figures/amblation5.eps', pad_inches = 0 , bbox_inche='tight')
    
    plt.show()

def plot_e2e_delay(roi_delay, nofilter_delay, nomerge_delay, noroi_delay, baseline_delay):
    org_roi_delay = np.cumsum(roi_delay[:,0])
    org_nofilter_delay = np.cumsum(nofilter_delay[:,0])
    org_nomerge_delay = np.cumsum(nomerge_delay[:,0])
    org_noroi_delay = np.cumsum(noroi_delay[:,0])
    org_baseline_delay = np.cumsum(baseline_delay[:,0])
    print('====', org_baseline_delay)
    roi_delay = np.sum(roi_delay, axis=0)
    nofilter_delay = np.sum(nofilter_delay, axis=0)
    nomerge_delay = np.sum(nomerge_delay, axis=0)
    noroi_delay = np.sum(noroi_delay, axis=0)
    baseline_delay = np.sum(baseline_delay, axis=0)
    
    width = 0.4
    fig, ax = plt.subplots()
    labels = ['CrossRoI', 'No-Filters', 'No-RoIInf', 'No-Merging', 'Baseline']
    x = np.arange(len(labels))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)

    roi_err = np.abs([roi_delay[0]-roi_delay[1], roi_delay[0]-roi_delay[2]]).reshape(2,1)
    rect0 = ax.bar( 0, roi_delay[0], yerr=roi_err, width=width, label='Server Delay',\
            edgecolor='black', color=delay_colors['Server'])

    nofilter_err = np.abs([nofilter_delay[0]-nofilter_delay[1], nofilter_delay[0]-nofilter_delay[2]]).reshape(2,1)
    rect2 = ax.bar(1, nofilter_delay[0], yerr=nofilter_err, width=width,\
            edgecolor='black', color=delay_colors['Server'])

    noroi_err = np.abs([noroi_delay[0]-noroi_delay[1], noroi_delay[0]-noroi_delay[2]]).reshape(2,1)
    rect1 = ax.bar(2, noroi_delay[0], yerr=noroi_err, width=width,\
            edgecolor='black', color=delay_colors['Server'])

    nomerge_err = np.abs([nomerge_delay[0]-nomerge_delay[1], nomerge_delay[0]-nomerge_delay[2]]).reshape(2,1)
    rect3 = ax.bar(3, nomerge_delay[0], yerr=nomerge_err, width=width,\
            edgecolor='black', color=delay_colors['Server'])

    baseline_err = np.abs([baseline_delay[0]-baseline_delay[1], baseline_delay[0]-baseline_delay[2]]).reshape(2,1)
    rect4 = ax.bar(4, baseline_delay[0], yerr=baseline_err, width=width,\
            edgecolor='black', color=delay_colors['Server'])

    ax.bar(0, org_roi_delay[1], width=width, edgecolor='black', label='Network Delay',color=delay_colors['Network'], hatch=anblation_bar_style['No-RoIInf'])
    ax.bar(1, org_nofilter_delay[1],  width=width,edgecolor='black', color=delay_colors['Network'], hatch=anblation_bar_style['No-RoIInf'])
    ax.bar(2, org_noroi_delay[1],  width=width,edgecolor='black', color=delay_colors['Network'], hatch=anblation_bar_style['No-RoIInf'])
    ax.bar(3, org_nomerge_delay[1],  width=width,edgecolor='black', color=delay_colors['Network'], hatch=anblation_bar_style['No-RoIInf'])
    ax.bar(4, org_baseline_delay[1], width=width,edgecolor='black', color=delay_colors['Network'], hatch=anblation_bar_style['No-RoIInf'])
    
    ax.bar(0, org_roi_delay[0], width=width, edgecolor='black', label='Camera Delay', color=delay_colors['Camera'], hatch=anblation_bar_style['No-Filters'])
    ax.bar(1, org_nofilter_delay[0],  width=width,edgecolor='black', color=delay_colors['Camera'], hatch=anblation_bar_style['No-Filters'])
    ax.bar(2, org_noroi_delay[0],  width=width,edgecolor='black', color=delay_colors['Camera'], hatch=anblation_bar_style['No-Filters'])
    ax.bar(3, org_nomerge_delay[0],  width=width,edgecolor='black', color=delay_colors['Camera'], hatch=anblation_bar_style['No-Filters'])
    ax.bar(4, org_baseline_delay[0], width=width,edgecolor='black', color=delay_colors['Camera'], hatch=anblation_bar_style['No-Filters'])
  
    ax.yaxis.grid(linestyle='--', which='major', linewidth=1.5)
    ax.set_ylabel('End to End Delay (s)')
    ax.legend(bbox_to_anchor=(0.58,1.02), ncol=1)
    # ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
    #             mode="expand", borderaxespad=0, ncol=2)

    ax.set_ylim(0, 3.5)

    autolabel(rect0, ax)
    autolabel(rect1, ax)
    autolabel(rect2, ax)
    autolabel(rect3, ax)
    autolabel(rect4, ax)

    ratio = 0.6
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    ax.set_aspect(1.0/ax.get_data_ratio()*ratio)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig('figures/amblation6.eps', pad_inches = 0, bbox_inche='tight')
    
    plt.show()

if __name__ == '__main__':

    cameras = general.cameras


    ## For Amblation Accuracy Evaluation
    accuracy_dir = 'results/S01/accuracy/'
    baseline_acc = np.load(accuracy_dir + 'baseline.npy')
    roi_acc = np.load(accuracy_dir + 'roi/' + '2e-05_0.01'+'.npy')
    nofilter_acc = np.load(accuracy_dir + 'roi/' + 'nofilter'+'.npy')
    # plot_count_baseline_roi_nofilter(baseline_arr, nofilter_arr, nofilter_arr, [300, 1500])
    ax1 = plot_acc_bar_roi_nofilter(baseline_acc, roi_acc, nofilter_acc, [300, 1500])
    ax2 = plot_err_dist_bar_roi_nofilter(baseline_acc, roi_acc, nofilter_acc, [300, 1500])
    

    ## For Amblation Network Evaluation
    network_dir = 'results/S01/network/results.npy'
    net_res = np.load(network_dir, allow_pickle=True).item()
    seg_amplyfy = {cam: {ck: 1 for ck in [1, 1.5, 2, 2.5, 3, 10]} for cam in cameras}
    for cam in cameras:
        for ck in [1, 1.5, 2, 2.5, 3, 10]:
            seg_amplyfy[cam][ck] = net_res['segments'][(cam, ck)]['avg_bitrate'] / \
                                   net_res['segments'][(cam, 10)]['avg_bitrate']
    baseline_net = []
    for cam in cameras:
        baseline_net.append( np.array(net_res['baseline'][(cam, 1)]['bitrate_per_chunk'][30:150]) * seg_amplyfy[cam][1])
    baseline_net = np.array(baseline_net)
    # baseline_net = baseline_net.sum(axis=0)
    # print('baseline_net', np.mean(baseline_net))

    roi_net = []
    for cam in cameras:
        roi_net.append( np.array(net_res['2e-05_0.01'][(cam, 1)]['bitrate_per_chunk'][30:150]) * seg_amplyfy[cam][1])
    roi_net = np.array(roi_net)
    # roi_net = roi_net.sum(axis=0)
    # print('roi_net', np.mean(roi_net))

    nofilter_net = []
    for cam in cameras:
        nofilter_net.append( np.array(net_res['nofilter'][(cam, 1)]['bitrate_per_chunk'][30:150]) * seg_amplyfy[cam][1])
    nofilter_net = np.array(nofilter_net)
    # nofilter_net = nofilter_net.sum(axis=0)
    # print('nofilter_net', np.mean(nofilter_net))

    nomerge_net = []
    for cam in cameras:
        nomerge_net.append( np.array(net_res['2e-05_0.01'][(cam, 1)]['bitrate_per_chunk'][30:150]) * seg_amplyfy[cam][1] * net_res['nomerge'][cam]['amplification'])
    nomerge_net = np.array(nomerge_net)
    nomerge_net_print = nomerge_net.sum(axis=0)
    print('nomerge_net', np.mean(nomerge_net_print))

    for setting in general.experiment_subdirs + ['baseline']:
        queue = []
        for cam in cameras:
            queue.append( np.array(net_res[setting][(cam, 1)]['bitrate_per_chunk'][30:150]) * seg_amplyfy[cam][1] )
        queue = np.array(queue)
        queue = queue.sum(axis=0)
        print(setting, np.mean(queue))

    ax3 = plot_network_bar_baseline_roi_nofilter_nomerge(baseline_net, roi_net, nofilter_net, nomerge_net, [300, 1500])

    ## For Amblation GPU Throughput Evaluation

    server_speed_dir = 'results/S01/server_speed/'
    server_res = np.load(server_speed_dir + 'speed_5.npy', allow_pickle=True).item()

    noroi_server = []
    for cam in cameras:
        noroi_server.append( np.mean(np.array(server_res['baseline'][cam] )))
    noroi_server = np.array(noroi_server)

    roi_server = []
    for cam in cameras:
        roi_server.append( np.mean(np.array(server_res['2e-05_0.01'][cam] )))
    roi_server = np.array(roi_server)

    nofilter_server = []
    for cam in cameras:
        nofilter_server.append( np.mean(np.array(server_res['nofilter'][cam] )))
    nofilter_server = np.array(nofilter_server)

    for i in range(len(roi_server)):
        if np.mean(roi_server[i]) < np.mean(noroi_server[i]):
            roi_server[i] = noroi_server[i]

    # noroi_server = np.mean(noroi_server, axis=0)
    # roi_server = np.mean(roi_server, axis=0)
    # nofilter_server = np.mean(nofilter_server, axis=0)

    for setting in general.experiment_subdirs + ['baseline']:
        queue = []
        for cam in cameras:
            queue.append(np.mean(np.array(server_res[setting][cam])))
        print(setting, np.mean(queue))

    ax4 = plot_server_bar_noroi_roi_nofilter_baseline(noroi_server, roi_server, nofilter_server)


    ## For Amblation Camera Throughput Evaluation
    camera_speed_dir = 'results/S01/camera_speed/'
    camera_res = np.load(camera_speed_dir + 'result.npy', allow_pickle=True).item()

    baseline_camera = []
    for cam in cameras:
        baseline_camera.append( np.mean(np.array(camera_res['baseline'][cam] )))
    baseline_camera = np.array(baseline_camera)

    roi_camera = []
    for cam in cameras:
        roi_camera.append( np.mean(np.array(camera_res['2e-05_0.01'][cam] )))
    roi_camera = np.array(roi_camera)

    nofilter_camera = []
    for cam in cameras:
        nofilter_camera.append( np.mean(np.array(camera_res['nofilter'][cam] )))
    nofilter_camera = np.array(nofilter_camera)

    for setting in general.experiment_subdirs + ['baseline']:
        queue = []
        for cam in cameras:
            queue.append(np.mean(np.array(camera_res[setting][cam])))
        print(setting, np.mean(queue))

    ax5 = plot_camera_bar_nomerge_roi_nofilter_baseline(roi_camera, nofilter_camera, baseline_camera)

    # Camera Delay
    baseline_cam_delay = [0.5 + 0.5/np.mean(baseline_camera), \
                          0.5 + 0.5/np.percentile(baseline_camera, 75), \
                          0.5 + 0.5/np.percentile(baseline_camera, 25)]  

    roi_cam_delay = [0.5 + 0.5/np.mean(roi_camera), \
                          0.5 + 0.5/np.percentile(roi_camera, 75), \
                          0.5 + 0.5/np.percentile(roi_camera, 25)]
                            
    nofilter_cam_delay = [0.5 + 0.5/np.mean(nofilter_camera), \
                          0.5 + 0.5/np.percentile(nofilter_camera, 75), \
                          0.5 + 0.5/np.percentile(nofilter_camera, 25)]
                            
    nomerge_cam_delay = [0.5 + 0.5/np.mean(roi_camera), \
                          0.5 + 0.5/np.percentile(roi_camera, 75), \
                          0.5 + 0.5/np.percentile(roi_camera, 25)]

    noroi_cam_delay = [0.5 + 0.5/np.mean(roi_camera), \
                        0.5 + 0.5/np.percentile(roi_camera, 75), \
                        0.5 + 0.5/np.percentile(roi_camera, 25)]

    # Server Delay
    baseline_server_delay = [25 / 2/ np.mean(noroi_server), \
                          25 / 2/ np.percentile(noroi_server, 75), \
                          25 / 2/ np.percentile(noroi_server, 25)]  

    roi_server_delay = [25 / 2 / np.mean(roi_server), \
                          25 / 2/ np.percentile(roi_server, 75), \
                          25 / 2/ np.percentile(roi_server, 25)]
                            
    nofilter_server_delay = [25 / 2/ np.mean(nofilter_server), \
                          25 / 2/ np.percentile(nofilter_server, 75), \
                          25 / 2/ np.percentile(nofilter_server, 25)]
                            
    nomerge_server_delay = [25 / 2/ np.mean(roi_server), \
                          25 / 2/ np.percentile(roi_server, 75), \
                          25 / 2/ np.percentile(roi_server, 25)]
                            
    noroi_server_delay = [25 / 2/ np.mean(noroi_server), \
                          25 / 2/ np.percentile(noroi_server, 75), \
                          25 / 2/ np.percentile(noroi_server, 25)]

    # Network Delay
    baseline_net = np.sum(baseline_net, axis=0)
    roi_net = np.sum(roi_net, axis=0)
    nofilter_net = np.sum(nofilter_net, axis=0)
    nomerge_net = np.sum(nomerge_net, axis=0)
    noroi_net = np.sum(roi_net, axis=0)

    baseline_network_delay = [0.05 + np.mean(baseline_net) / 30000, \
                          0.05 + np.percentile(baseline_net, 75) / 30000, \
                          0.05 + np.percentile(baseline_net, 25) / 30000]  

    roi_network_delay = [0.05 +  np.mean(roi_net) / 30000, \
                          0.05 +  np.percentile(roi_net, 75) / 30000, \
                          0.05 +  np.percentile(roi_net, 25) / 30000]
                            
    nofilter_network_delay = [0.05 +  np.mean(nofilter_net) / 30000, \
                          0.05 +  np.percentile(nofilter_net, 75) / 30000, \
                          0.05 +  np.percentile(nofilter_net, 25) / 30000]
                            
    nomerge_network_delay = [0.05 +  np.mean(nomerge_net) / 30000, \
                          0.05 +  np.percentile(nomerge_net, 75) / 30000, \
                          0.05 +  np.percentile(nomerge_net, 25) / 30000]
                            
    noroi_network_delay = [0.05 +  np.mean(roi_net) / 30000,\
                          0.05 +  np.percentile(roi_net, 75) / 30000, \
                          0.05 +  np.percentile(roi_net, 25) / 30000]

    baseline_delay = np.array([baseline_cam_delay, baseline_network_delay, baseline_server_delay])
    roi_delay = np.array([roi_cam_delay, roi_network_delay, roi_server_delay])
    nofilter_delay = np.array([nofilter_cam_delay, nofilter_network_delay, nofilter_server_delay])
    nomerge_delay = np.array([nomerge_cam_delay, nomerge_network_delay, nomerge_server_delay])
    noroi_delay = np.array([noroi_cam_delay, noroi_network_delay, noroi_server_delay])

    plot_e2e_delay(roi_delay, nofilter_delay, nomerge_delay, noroi_delay, baseline_delay)
    
    print(baseline_cam_delay)
    print(baseline_network_delay)
    print(baseline_server_delay)

    print(roi_cam_delay)
    print(roi_network_delay)
    print(roi_server_delay)