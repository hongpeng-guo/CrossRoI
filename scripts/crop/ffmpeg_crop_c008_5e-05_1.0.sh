ffmpeg -i /home/hongpeng/Desktop/research/AICity/Track3/validation/S02/c008/vdo.avi -filter_complex "[0]split=2[s0][s1]; [s0]crop=1920:960:0:128[s0]; [s1]crop=1472:64:0:64[s1]" -map [s0] -c:v libx264 videos/S02/5e-05_1.0/tmp/c008_00.mp4 -map [s1] -c:v libx264 videos/S02/5e-05_1.0/tmp/c008_01.mp4 