ffmpeg -i videos/S02/5e-06_1.0/tmp/c007_00.mp4 -i videos/S02/5e-06_1.0/tmp/c007_01.mp4 -i videos/S02/5e-06_1.0/tmp/c007_02.mp4 -i videos/S02/5e-06_1.0/tmp/c007_03.mp4 -i videos/S02/5e-06_1.0/tmp/c007_04.mp4 -i videos/S02/5e-06_1.0/tmp/c007_05.mp4 -i videos/S02/5e-06_1.0/tmp/c007_06.mp4 -i videos/S02/5e-06_1.0/tmp/c007_07.mp4 -i videos/S02/5e-06_1.0/tmp/c007_08.mp4 -filter_complex "nullsrc=size=1920x1080 [base]; [0:v] setpts=PTS-STARTPTS [s0]; [1:v] setpts=PTS-STARTPTS [s1]; [2:v] setpts=PTS-STARTPTS [s2]; [3:v] setpts=PTS-STARTPTS [s3]; [4:v] setpts=PTS-STARTPTS [s4]; [5:v] setpts=PTS-STARTPTS [s5]; [6:v] setpts=PTS-STARTPTS [s6]; [7:v] setpts=PTS-STARTPTS [s7]; [8:v] setpts=PTS-STARTPTS [s8]; [base][s0] overlay=shortest=1:x=384:y=128 [tmp0]; [tmp0][s1] overlay=shortest=1:x=0:y=128 [tmp1]; [tmp1][s2] overlay=shortest=1:x=1216:y=896 [tmp2]; [tmp2][s3] overlay=shortest=1:x=320:y=896 [tmp3]; [tmp3][s4] overlay=shortest=1:x=704:y=64 [tmp4]; [tmp4][s5] overlay=shortest=1:x=320:y=640 [tmp5]; [tmp5][s6] overlay=shortest=1:x=1024:y=896 [tmp6]; [tmp6][s7] overlay=shortest=1:x=832:y=896 [tmp7]; [tmp7][s8] overlay=shortest=1:x=1152:y=960 " -c:v libx264 -r 10 videos/S02/5e-06_1.0/croped_c007.mp4 