ffmpeg -i videos/S02/2e-05_0.1/tmp/c006_00.mp4 -i videos/S02/2e-05_0.1/tmp/c006_01.mp4 -i videos/S02/2e-05_0.1/tmp/c006_02.mp4 -i videos/S02/2e-05_0.1/tmp/c006_03.mp4 -i videos/S02/2e-05_0.1/tmp/c006_04.mp4 -i videos/S02/2e-05_0.1/tmp/c006_05.mp4 -i videos/S02/2e-05_0.1/tmp/c006_06.mp4 -i videos/S02/2e-05_0.1/tmp/c006_07.mp4 -i videos/S02/2e-05_0.1/tmp/c006_08.mp4 -i videos/S02/2e-05_0.1/tmp/c006_09.mp4 -i videos/S02/2e-05_0.1/tmp/c006_10.mp4 -i videos/S02/2e-05_0.1/tmp/c006_11.mp4 -i videos/S02/2e-05_0.1/tmp/c006_12.mp4 -i videos/S02/2e-05_0.1/tmp/c006_13.mp4 -i videos/S02/2e-05_0.1/tmp/c006_14.mp4 -i videos/S02/2e-05_0.1/tmp/c006_15.mp4 -i videos/S02/2e-05_0.1/tmp/c006_16.mp4 -i videos/S02/2e-05_0.1/tmp/c006_17.mp4 -filter_complex "nullsrc=size=1920x1080 [base]; [0:v] setpts=PTS-STARTPTS [s0]; [1:v] setpts=PTS-STARTPTS [s1]; [2:v] setpts=PTS-STARTPTS [s2]; [3:v] setpts=PTS-STARTPTS [s3]; [4:v] setpts=PTS-STARTPTS [s4]; [5:v] setpts=PTS-STARTPTS [s5]; [6:v] setpts=PTS-STARTPTS [s6]; [7:v] setpts=PTS-STARTPTS [s7]; [8:v] setpts=PTS-STARTPTS [s8]; [9:v] setpts=PTS-STARTPTS [s9]; [10:v] setpts=PTS-STARTPTS [s10]; [11:v] setpts=PTS-STARTPTS [s11]; [12:v] setpts=PTS-STARTPTS [s12]; [13:v] setpts=PTS-STARTPTS [s13]; [14:v] setpts=PTS-STARTPTS [s14]; [15:v] setpts=PTS-STARTPTS [s15]; [16:v] setpts=PTS-STARTPTS [s16]; [17:v] setpts=PTS-STARTPTS [s17]; [base][s0] overlay=shortest=1:x=0:y=512 [tmp0]; [tmp0][s1] overlay=shortest=1:x=64:y=128 [tmp1]; [tmp1][s2] overlay=shortest=1:x=1344:y=64 [tmp2]; [tmp2][s3] overlay=shortest=1:x=0:y=832 [tmp3]; [tmp3][s4] overlay=shortest=1:x=1536:y=384 [tmp4]; [tmp4][s5] overlay=shortest=1:x=384:y=64 [tmp5]; [tmp5][s6] overlay=shortest=1:x=1216:y=0 [tmp6]; [tmp6][s7] overlay=shortest=1:x=448:y=832 [tmp7]; [tmp7][s8] overlay=shortest=1:x=896:y=128 [tmp8]; [tmp8][s9] overlay=shortest=1:x=0:y=192 [tmp9]; [tmp9][s10] overlay=shortest=1:x=768:y=0 [tmp10]; [tmp10][s11] overlay=shortest=1:x=1856:y=128 [tmp11]; [tmp11][s12] overlay=shortest=1:x=960:y=128 [tmp12]; [tmp12][s13] overlay=shortest=1:x=1344:y=0 [tmp13]; [tmp13][s14] overlay=shortest=1:x=960:y=192 [tmp14]; [tmp14][s15] overlay=shortest=1:x=1280:y=256 [tmp15]; [tmp15][s16] overlay=shortest=1:x=1472:y=384 [tmp16]; [tmp16][s17] overlay=shortest=1:x=448:y=960 " -c:v libx264 -r 10 videos/S02/2e-05_0.1/croped_c006.mp4 