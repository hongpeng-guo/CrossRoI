ffmpeg -i /home/hongpeng/Desktop/research/AICity/Track3/validation/S02/c006/vdo.avi -filter_complex "[0]split=13[s0][s1][s2][s3][s4][s5][s6][s7][s8][s9][s10][s11][s12]; [s0]crop=704:768:0:192[s0]; [s1]crop=384:704:1536:128[s1]; [s2]crop=704:128:384:64[s2]; [s3]crop=256:320:1280:512[s3]; [s4]crop=256:256:1280:64[s4]; [s5]crop=192:320:704:192[s5]; [s6]crop=320:64:1536:64[s6]; [s7]crop=192:64:1216:0[s7]; [s8]crop=192:64:1344:320[s8]; [s9]crop=64:128:1216:64[s9]; [s10]crop=128:64:896:192[s10]; [s11]crop=64:64:896:256[s11]; [s12]crop=64:64:1472:384[s12]" -map [s0] -c:v libx264 videos/S02/5e-06_1.0/tmp/c006_00.mp4 -map [s1] -c:v libx264 videos/S02/5e-06_1.0/tmp/c006_01.mp4 -map [s2] -c:v libx264 videos/S02/5e-06_1.0/tmp/c006_02.mp4 -map [s3] -c:v libx264 videos/S02/5e-06_1.0/tmp/c006_03.mp4 -map [s4] -c:v libx264 videos/S02/5e-06_1.0/tmp/c006_04.mp4 -map [s5] -c:v libx264 videos/S02/5e-06_1.0/tmp/c006_05.mp4 -map [s6] -c:v libx264 videos/S02/5e-06_1.0/tmp/c006_06.mp4 -map [s7] -c:v libx264 videos/S02/5e-06_1.0/tmp/c006_07.mp4 -map [s8] -c:v libx264 videos/S02/5e-06_1.0/tmp/c006_08.mp4 -map [s9] -c:v libx264 videos/S02/5e-06_1.0/tmp/c006_09.mp4 -map [s10] -c:v libx264 videos/S02/5e-06_1.0/tmp/c006_10.mp4 -map [s11] -c:v libx264 videos/S02/5e-06_1.0/tmp/c006_11.mp4 -map [s12] -c:v libx264 videos/S02/5e-06_1.0/tmp/c006_12.mp4 