ffmpeg -i /home/hongpeng/Desktop/research/AICity/Track3/validation/S02/c006/vdo.avi -filter_complex "[0]split=16[s0][s1][s2][s3][s4][s5][s6][s7][s8][s9][s10][s11][s12][s13][s14][s15]; [s0]crop=1920:320:0:512[s0]; [s1]crop=896:320:0:192[s1]; [s2]crop=512:320:1344:64[s2]; [s3]crop=448:256:0:832[s3]; [s4]crop=704:128:384:64[s4]; [s5]crop=384:128:1536:384[s5]; [s6]crop=128:256:1216:0[s6]; [s7]crop=256:128:448:832[s7]; [s8]crop=320:64:768:0[s8]; [s9]crop=64:256:1856:128[s9]; [s10]crop=64:256:896:192[s10]; [s11]crop=64:64:1344:0[s11]; [s12]crop=64:64:960:192[s12]; [s13]crop=64:64:1280:256[s13]; [s14]crop=64:64:1472:384[s14]; [s15]crop=64:64:448:960[s15]" -map [s0] -c:v libx264 videos/S02/2e-05_10.0/tmp/c006_00.mp4 -map [s1] -c:v libx264 videos/S02/2e-05_10.0/tmp/c006_01.mp4 -map [s2] -c:v libx264 videos/S02/2e-05_10.0/tmp/c006_02.mp4 -map [s3] -c:v libx264 videos/S02/2e-05_10.0/tmp/c006_03.mp4 -map [s4] -c:v libx264 videos/S02/2e-05_10.0/tmp/c006_04.mp4 -map [s5] -c:v libx264 videos/S02/2e-05_10.0/tmp/c006_05.mp4 -map [s6] -c:v libx264 videos/S02/2e-05_10.0/tmp/c006_06.mp4 -map [s7] -c:v libx264 videos/S02/2e-05_10.0/tmp/c006_07.mp4 -map [s8] -c:v libx264 videos/S02/2e-05_10.0/tmp/c006_08.mp4 -map [s9] -c:v libx264 videos/S02/2e-05_10.0/tmp/c006_09.mp4 -map [s10] -c:v libx264 videos/S02/2e-05_10.0/tmp/c006_10.mp4 -map [s11] -c:v libx264 videos/S02/2e-05_10.0/tmp/c006_11.mp4 -map [s12] -c:v libx264 videos/S02/2e-05_10.0/tmp/c006_12.mp4 -map [s13] -c:v libx264 videos/S02/2e-05_10.0/tmp/c006_13.mp4 -map [s14] -c:v libx264 videos/S02/2e-05_10.0/tmp/c006_14.mp4 -map [s15] -c:v libx264 videos/S02/2e-05_10.0/tmp/c006_15.mp4 