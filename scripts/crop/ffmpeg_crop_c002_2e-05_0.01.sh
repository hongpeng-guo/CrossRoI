ffmpeg -i /home/hongpeng/Desktop/research/AICity/Track3/train/S01/c002/vdo.avi -filter_complex "[0]split=7[s0][s1][s2][s3][s4][s5][s6]; [s0]crop=1920:256:0:256[s0]; [s1]crop=768:384:960:704[s1]; [s2]crop=512:192:704:64[s2]; [s3]crop=768:128:704:512[s3]; [s4]crop=448:64:1472:512[s4]; [s5]crop=128:128:1216:128[s5]; [s6]crop=64:64:1344:192[s6]" -map [s0] -c:v libx264 videos/S01/2e-05_0.01/tmp/c002_00.mp4 -map [s1] -c:v libx264 videos/S01/2e-05_0.01/tmp/c002_01.mp4 -map [s2] -c:v libx264 videos/S01/2e-05_0.01/tmp/c002_02.mp4 -map [s3] -c:v libx264 videos/S01/2e-05_0.01/tmp/c002_03.mp4 -map [s4] -c:v libx264 videos/S01/2e-05_0.01/tmp/c002_04.mp4 -map [s5] -c:v libx264 videos/S01/2e-05_0.01/tmp/c002_05.mp4 -map [s6] -c:v libx264 videos/S01/2e-05_0.01/tmp/c002_06.mp4 