ffmpeg -i /home/hongpeng/Desktop/research/AICity/Track3/validation/S02/c007/vdo.avi -filter_complex "[0]split=9[s0][s1][s2][s3][s4][s5][s6][s7][s8]; [s0]crop=1536:768:384:128[s0]; [s1]crop=384:448:0:128[s1]; [s2]crop=704:192:1216:896[s2]; [s3]crop=512:192:320:896[s3]; [s4]crop=1216:64:704:64[s4]; [s5]crop=64:256:320:640[s5]; [s6]crop=192:64:1024:896[s6]; [s7]crop=64:64:832:896[s7]; [s8]crop=64:64:1152:960[s8]" -map [s0] -c:v libx264 videos/S02/5e-06_1.0/tmp/c007_00.mp4 -map [s1] -c:v libx264 videos/S02/5e-06_1.0/tmp/c007_01.mp4 -map [s2] -c:v libx264 videos/S02/5e-06_1.0/tmp/c007_02.mp4 -map [s3] -c:v libx264 videos/S02/5e-06_1.0/tmp/c007_03.mp4 -map [s4] -c:v libx264 videos/S02/5e-06_1.0/tmp/c007_04.mp4 -map [s5] -c:v libx264 videos/S02/5e-06_1.0/tmp/c007_05.mp4 -map [s6] -c:v libx264 videos/S02/5e-06_1.0/tmp/c007_06.mp4 -map [s7] -c:v libx264 videos/S02/5e-06_1.0/tmp/c007_07.mp4 -map [s8] -c:v libx264 videos/S02/5e-06_1.0/tmp/c007_08.mp4 