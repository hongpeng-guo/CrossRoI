ffmpeg -i /home/hongpeng/Desktop/research/AICity/Track3/validation/S02/c009/vdo.avi -filter_complex "[0]split=13[s0][s1][s2][s3][s4][s5][s6][s7][s8][s9][s10][s11][s12]; [s0]crop=1088:832:0:192[s0]; [s1]crop=704:448:1088:512[s1]; [s2]crop=512:256:1088:128[s2]; [s3]crop=1216:64:128:1024[s3]; [s4]crop=320:128:768:64[s4]; [s5]crop=128:320:1792:512[s5]; [s6]crop=256:128:1600:128[s6]; [s7]crop=512:64:1088:960[s7]; [s8]crop=320:64:1280:384[s8]; [s9]crop=256:64:448:128[s9]; [s10]crop=128:64:576:64[s10]; [s11]crop=64:64:1856:192[s11]; [s12]crop=64:64:1792:832[s12]" -map [s0] -c:v libx264 videos/S02/1e-05_1.0/tmp/c009_00.mp4 -map [s1] -c:v libx264 videos/S02/1e-05_1.0/tmp/c009_01.mp4 -map [s2] -c:v libx264 videos/S02/1e-05_1.0/tmp/c009_02.mp4 -map [s3] -c:v libx264 videos/S02/1e-05_1.0/tmp/c009_03.mp4 -map [s4] -c:v libx264 videos/S02/1e-05_1.0/tmp/c009_04.mp4 -map [s5] -c:v libx264 videos/S02/1e-05_1.0/tmp/c009_05.mp4 -map [s6] -c:v libx264 videos/S02/1e-05_1.0/tmp/c009_06.mp4 -map [s7] -c:v libx264 videos/S02/1e-05_1.0/tmp/c009_07.mp4 -map [s8] -c:v libx264 videos/S02/1e-05_1.0/tmp/c009_08.mp4 -map [s9] -c:v libx264 videos/S02/1e-05_1.0/tmp/c009_09.mp4 -map [s10] -c:v libx264 videos/S02/1e-05_1.0/tmp/c009_10.mp4 -map [s11] -c:v libx264 videos/S02/1e-05_1.0/tmp/c009_11.mp4 -map [s12] -c:v libx264 videos/S02/1e-05_1.0/tmp/c009_12.mp4 