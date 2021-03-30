ffmpeg -i videos/S01/2e-05_1.0/tmp/c002_00.mp4 -i videos/S01/2e-05_1.0/tmp/c002_01.mp4 -i videos/S01/2e-05_1.0/tmp/c002_02.mp4 -i videos/S01/2e-05_1.0/tmp/c002_03.mp4 -i videos/S01/2e-05_1.0/tmp/c002_04.mp4 -i videos/S01/2e-05_1.0/tmp/c002_05.mp4 -i videos/S01/2e-05_1.0/tmp/c002_06.mp4 -filter_complex "nullsrc=size=1920x1080 [base]; [0:v] setpts=PTS-STARTPTS [s0]; [1:v] setpts=PTS-STARTPTS [s1]; [2:v] setpts=PTS-STARTPTS [s2]; [3:v] setpts=PTS-STARTPTS [s3]; [4:v] setpts=PTS-STARTPTS [s4]; [5:v] setpts=PTS-STARTPTS [s5]; [6:v] setpts=PTS-STARTPTS [s6]; [base][s0] overlay=shortest=1:x=576:y=384 [tmp0]; [tmp0][s1] overlay=shortest=1:x=64:y=384 [tmp1]; [tmp1][s2] overlay=shortest=1:x=768:y=192 [tmp2]; [tmp2][s3] overlay=shortest=1:x=704:y=64 [tmp3]; [tmp3][s4] overlay=shortest=1:x=1600:y=256 [tmp4]; [tmp4][s5] overlay=shortest=1:x=1792:y=384 [tmp5]; [tmp5][s6] overlay=shortest=1:x=1216:y=128 " -c:v libx264 -r 10 videos/S01/2e-05_1.0/croped_c002.mp4 