ffmpeg -i videos/S02/2e-05_10.0/tmp/c008_00.mp4 -i videos/S02/2e-05_10.0/tmp/c008_01.mp4 -filter_complex "nullsrc=size=1920x1080 [base]; [0:v] setpts=PTS-STARTPTS [s0]; [1:v] setpts=PTS-STARTPTS [s1]; [base][s0] overlay=shortest=1:x=0:y=128 [tmp0]; [tmp0][s1] overlay=shortest=1:x=0:y=64 " -c:v libx264 -r 10 videos/S02/2e-05_10.0/croped_c008.mp4 