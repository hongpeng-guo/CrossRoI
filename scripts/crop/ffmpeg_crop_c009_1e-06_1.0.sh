ffmpeg -i /home/hongpeng/Desktop/research/AICity/Track3/validation/S02/c009/vdo.avi -filter_complex "[0]split=21[s0][s1][s2][s3][s4][s5][s6][s7][s8][s9][s10][s11][s12][s13][s14][s15][s16][s17][s18][s19][s20]; [s0]crop=1472:512:128:512[s0]; [s1]crop=768:320:0:192[s1]; [s2]crop=640:128:960:256[s2]; [s3]crop=1216:64:128:1024[s3]; [s4]crop=192:384:1600:576[s4]; [s5]crop=448:128:832:128[s5]; [s6]crop=256:192:1664:320[s6]; [s7]crop=128:384:0:512[s7]; [s8]crop=320:128:1536:128[s8]; [s9]crop=320:128:768:384[s9]; [s10]crop=320:64:768:64[s10]; [s11]crop=320:64:1280:384[s11]; [s12]crop=128:128:576:64[s12]; [s13]crop=64:128:768:256[s13]; [s14]crop=64:64:1472:128[s14]; [s15]crop=64:64:768:128[s15]; [s16]crop=64:64:512:128[s16]; [s17]crop=64:64:1856:192[s17]; [s18]crop=64:64:1280:192[s18]; [s19]crop=64:64:896:256[s19]; [s20]crop=64:64:1600:512[s20]" -map [s0] -c:v libx264 videos/S02/1e-06_1.0/tmp/c009_00.mp4 -map [s1] -c:v libx264 videos/S02/1e-06_1.0/tmp/c009_01.mp4 -map [s2] -c:v libx264 videos/S02/1e-06_1.0/tmp/c009_02.mp4 -map [s3] -c:v libx264 videos/S02/1e-06_1.0/tmp/c009_03.mp4 -map [s4] -c:v libx264 videos/S02/1e-06_1.0/tmp/c009_04.mp4 -map [s5] -c:v libx264 videos/S02/1e-06_1.0/tmp/c009_05.mp4 -map [s6] -c:v libx264 videos/S02/1e-06_1.0/tmp/c009_06.mp4 -map [s7] -c:v libx264 videos/S02/1e-06_1.0/tmp/c009_07.mp4 -map [s8] -c:v libx264 videos/S02/1e-06_1.0/tmp/c009_08.mp4 -map [s9] -c:v libx264 videos/S02/1e-06_1.0/tmp/c009_09.mp4 -map [s10] -c:v libx264 videos/S02/1e-06_1.0/tmp/c009_10.mp4 -map [s11] -c:v libx264 videos/S02/1e-06_1.0/tmp/c009_11.mp4 -map [s12] -c:v libx264 videos/S02/1e-06_1.0/tmp/c009_12.mp4 -map [s13] -c:v libx264 videos/S02/1e-06_1.0/tmp/c009_13.mp4 -map [s14] -c:v libx264 videos/S02/1e-06_1.0/tmp/c009_14.mp4 -map [s15] -c:v libx264 videos/S02/1e-06_1.0/tmp/c009_15.mp4 -map [s16] -c:v libx264 videos/S02/1e-06_1.0/tmp/c009_16.mp4 -map [s17] -c:v libx264 videos/S02/1e-06_1.0/tmp/c009_17.mp4 -map [s18] -c:v libx264 videos/S02/1e-06_1.0/tmp/c009_18.mp4 -map [s19] -c:v libx264 videos/S02/1e-06_1.0/tmp/c009_19.mp4 -map [s20] -c:v libx264 videos/S02/1e-06_1.0/tmp/c009_20.mp4 