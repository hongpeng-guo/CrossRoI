ffmpeg -i /home/hongpeng/Desktop/research/AICity/Track3/train/S01/c003/vdo.avi -filter_complex "[0]split=9[s0][s1][s2][s3][s4][s5][s6][s7][s8]; [s0]crop=1024:512:64:192[s0]; [s1]crop=704:384:1088:64[s1]; [s2]crop=576:128:512:64[s2]; [s3]crop=128:320:1792:128[s3]; [s4]crop=512:64:1088:448[s4]; [s5]crop=64:448:0:256[s5]; [s6]crop=128:128:1088:512[s6]; [s7]crop=128:64:384:128[s7]; [s8]crop=64:64:1216:512[s8]" -map [s0] -c:v libx264 videos/S01/2e-05_10.0/tmp/c003_00.mp4 -map [s1] -c:v libx264 videos/S01/2e-05_10.0/tmp/c003_01.mp4 -map [s2] -c:v libx264 videos/S01/2e-05_10.0/tmp/c003_02.mp4 -map [s3] -c:v libx264 videos/S01/2e-05_10.0/tmp/c003_03.mp4 -map [s4] -c:v libx264 videos/S01/2e-05_10.0/tmp/c003_04.mp4 -map [s5] -c:v libx264 videos/S01/2e-05_10.0/tmp/c003_05.mp4 -map [s6] -c:v libx264 videos/S01/2e-05_10.0/tmp/c003_06.mp4 -map [s7] -c:v libx264 videos/S01/2e-05_10.0/tmp/c003_07.mp4 -map [s8] -c:v libx264 videos/S01/2e-05_10.0/tmp/c003_08.mp4 