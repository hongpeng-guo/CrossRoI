ffmpeg -i /home/hongpeng/Desktop/research/AICity/Track3/validation/S02/c008/vdo.avi -filter_complex "[0]split=5[s0][s1][s2][s3][s4]; [s0]crop=1472:704:0:64[s0]; [s1]crop=448:448:1472:128[s1]; [s2]crop=448:192:128:768[s2]; [s3]crop=320:128:1472:576[s3]; [s4]crop=128:64:1472:704[s4]" -map [s0] -c:v libx264 videos/S02/1e-06_1.0/tmp/c008_00.mp4 -map [s1] -c:v libx264 videos/S02/1e-06_1.0/tmp/c008_01.mp4 -map [s2] -c:v libx264 videos/S02/1e-06_1.0/tmp/c008_02.mp4 -map [s3] -c:v libx264 videos/S02/1e-06_1.0/tmp/c008_03.mp4 -map [s4] -c:v libx264 videos/S02/1e-06_1.0/tmp/c008_04.mp4 