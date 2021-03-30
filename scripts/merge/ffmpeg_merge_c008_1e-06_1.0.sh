ffmpeg -i videos/S02/1e-06_1.0/tmp/c008_00.mp4 -i videos/S02/1e-06_1.0/tmp/c008_01.mp4 -i videos/S02/1e-06_1.0/tmp/c008_02.mp4 -i videos/S02/1e-06_1.0/tmp/c008_03.mp4 -i videos/S02/1e-06_1.0/tmp/c008_04.mp4 -filter_complex "nullsrc=size=1920x1080 [base]; [0:v] setpts=PTS-STARTPTS [s0]; [1:v] setpts=PTS-STARTPTS [s1]; [2:v] setpts=PTS-STARTPTS [s2]; [3:v] setpts=PTS-STARTPTS [s3]; [4:v] setpts=PTS-STARTPTS [s4]; [base][s0] overlay=shortest=1:x=0:y=64 [tmp0]; [tmp0][s1] overlay=shortest=1:x=1472:y=128 [tmp1]; [tmp1][s2] overlay=shortest=1:x=128:y=768 [tmp2]; [tmp2][s3] overlay=shortest=1:x=1472:y=576 [tmp3]; [tmp3][s4] overlay=shortest=1:x=1472:y=704 " -c:v libx264 -r 10 videos/S02/1e-06_1.0/croped_c008.mp4 