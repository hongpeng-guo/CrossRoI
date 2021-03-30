ffmpeg -i videos/S01/1e-05_1.0/tmp/c004_00.mp4 -i videos/S01/1e-05_1.0/tmp/c004_01.mp4 -i videos/S01/1e-05_1.0/tmp/c004_02.mp4 -i videos/S01/1e-05_1.0/tmp/c004_03.mp4 -i videos/S01/1e-05_1.0/tmp/c004_04.mp4 -i videos/S01/1e-05_1.0/tmp/c004_05.mp4 -filter_complex "nullsrc=size=1920x1080 [base]; [0:v] setpts=PTS-STARTPTS [s0]; [1:v] setpts=PTS-STARTPTS [s1]; [2:v] setpts=PTS-STARTPTS [s2]; [3:v] setpts=PTS-STARTPTS [s3]; [4:v] setpts=PTS-STARTPTS [s4]; [5:v] setpts=PTS-STARTPTS [s5]; [base][s0] overlay=shortest=1:x=0:y=256 [tmp0]; [tmp0][s1] overlay=shortest=1:x=0:y=640 [tmp1]; [tmp1][s2] overlay=shortest=1:x=320:y=128 [tmp2]; [tmp2][s3] overlay=shortest=1:x=192:y=1024 [tmp3]; [tmp3][s4] overlay=shortest=1:x=1088:y=192 [tmp4]; [tmp4][s5] overlay=shortest=1:x=128:y=192 " -c:v libx264 -r 10 videos/S01/1e-05_1.0/croped_c004.mp4 