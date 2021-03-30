ffmpeg -i videos/S02/1e-04_1.0/tmp/c009_00.mp4 -i videos/S02/1e-04_1.0/tmp/c009_01.mp4 -i videos/S02/1e-04_1.0/tmp/c009_02.mp4 -i videos/S02/1e-04_1.0/tmp/c009_03.mp4 -i videos/S02/1e-04_1.0/tmp/c009_04.mp4 -i videos/S02/1e-04_1.0/tmp/c009_05.mp4 -i videos/S02/1e-04_1.0/tmp/c009_06.mp4 -i videos/S02/1e-04_1.0/tmp/c009_07.mp4 -i videos/S02/1e-04_1.0/tmp/c009_08.mp4 -i videos/S02/1e-04_1.0/tmp/c009_09.mp4 -i videos/S02/1e-04_1.0/tmp/c009_10.mp4 -i videos/S02/1e-04_1.0/tmp/c009_11.mp4 -i videos/S02/1e-04_1.0/tmp/c009_12.mp4 -filter_complex "nullsrc=size=1920x1080 [base]; [0:v] setpts=PTS-STARTPTS [s0]; [1:v] setpts=PTS-STARTPTS [s1]; [2:v] setpts=PTS-STARTPTS [s2]; [3:v] setpts=PTS-STARTPTS [s3]; [4:v] setpts=PTS-STARTPTS [s4]; [5:v] setpts=PTS-STARTPTS [s5]; [6:v] setpts=PTS-STARTPTS [s6]; [7:v] setpts=PTS-STARTPTS [s7]; [8:v] setpts=PTS-STARTPTS [s8]; [9:v] setpts=PTS-STARTPTS [s9]; [10:v] setpts=PTS-STARTPTS [s10]; [11:v] setpts=PTS-STARTPTS [s11]; [12:v] setpts=PTS-STARTPTS [s12]; [base][s0] overlay=shortest=1:x=0:y=192 [tmp0]; [tmp0][s1] overlay=shortest=1:x=1088:y=512 [tmp1]; [tmp1][s2] overlay=shortest=1:x=1088:y=128 [tmp2]; [tmp2][s3] overlay=shortest=1:x=128:y=1024 [tmp3]; [tmp3][s4] overlay=shortest=1:x=1792:y=384 [tmp4]; [tmp4][s5] overlay=shortest=1:x=768:y=64 [tmp5]; [tmp5][s6] overlay=shortest=1:x=1536:y=384 [tmp6]; [tmp6][s7] overlay=shortest=1:x=1088:y=960 [tmp7]; [tmp7][s8] overlay=shortest=1:x=448:y=128 [tmp8]; [tmp8][s9] overlay=shortest=1:x=1856:y=192 [tmp9]; [tmp9][s10] overlay=shortest=1:x=1344:y=384 [tmp10]; [tmp10][s11] overlay=shortest=1:x=576:y=64 [tmp11]; [tmp11][s12] overlay=shortest=1:x=1792:y=832 " -c:v libx264 -r 10 videos/S02/1e-04_1.0/croped_c009.mp4 