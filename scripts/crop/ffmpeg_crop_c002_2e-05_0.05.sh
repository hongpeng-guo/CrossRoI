ffmpeg -i /home/hongpeng/Desktop/research/AICity/Track3/train/S01/c002/vdo.avi -filter_complex "[0]split=8[s0][s1][s2][s3][s4][s5][s6][s7]; [s0]crop=1856:320:64:384[s0]; [s1]crop=832:384:512:704[s1]; [s2]crop=1152:128:768:256[s2]; [s3]crop=448:192:768:64[s3]; [s4]crop=384:128:1344:704[s4]; [s5]crop=128:128:1216:128[s5]; [s6]crop=64:128:704:64[s6]; [s7]crop=64:64:1344:192[s7]" -map [s0] -c:v libx264 videos/S01/2e-05_0.05/tmp/c002_00.mp4 -map [s1] -c:v libx264 videos/S01/2e-05_0.05/tmp/c002_01.mp4 -map [s2] -c:v libx264 videos/S01/2e-05_0.05/tmp/c002_02.mp4 -map [s3] -c:v libx264 videos/S01/2e-05_0.05/tmp/c002_03.mp4 -map [s4] -c:v libx264 videos/S01/2e-05_0.05/tmp/c002_04.mp4 -map [s5] -c:v libx264 videos/S01/2e-05_0.05/tmp/c002_05.mp4 -map [s6] -c:v libx264 videos/S01/2e-05_0.05/tmp/c002_06.mp4 -map [s7] -c:v libx264 videos/S01/2e-05_0.05/tmp/c002_07.mp4 