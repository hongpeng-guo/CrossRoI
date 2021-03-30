ffmpeg -i videos/S01/2e-05_1.0/tmp/c005_00.mp4 -i videos/S01/2e-05_1.0/tmp/c005_01.mp4 -i videos/S01/2e-05_1.0/tmp/c005_02.mp4 -i videos/S01/2e-05_1.0/tmp/c005_03.mp4 -i videos/S01/2e-05_1.0/tmp/c005_04.mp4 -filter_complex "nullsrc=size=1280x960 [base]; [0:v] setpts=PTS-STARTPTS [s0]; [1:v] setpts=PTS-STARTPTS [s1]; [2:v] setpts=PTS-STARTPTS [s2]; [3:v] setpts=PTS-STARTPTS [s3]; [4:v] setpts=PTS-STARTPTS [s4]; [base][s0] overlay=shortest=1:x=64:y=192 [tmp0]; [tmp0][s1] overlay=shortest=1:x=384:y=832 [tmp1]; [tmp1][s2] overlay=shortest=1:x=64:y=128 [tmp2]; [tmp2][s3] overlay=shortest=1:x=0:y=192 [tmp3]; [tmp3][s4] overlay=shortest=1:x=256:y=832 " -c:v libx264 -r 10 videos/S01/2e-05_1.0/croped_c005.mp4 