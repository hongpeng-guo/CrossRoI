ffmpeg -i /home/hongpeng/Desktop/research/AICity/Track3/train/S01/c001/vdo.avi -filter_complex "[0]split=1[s0]; [s0]crop=1920:704:0:128[s0]" -map [s0] -c:v libx264 videos/S01/nofilter/tmp/c001_00.mp4 