ffmpeg -i videos/S01/nofilter/tmp/c001_00.mp4 -filter_complex "nullsrc=size=1920x1080 [base]; [0:v] setpts=PTS-STARTPTS [s0]; [base][s0] overlay=shortest=1:x=0:y=128 " -c:v libx264 -r 10 videos/S01/nofilter/croped_c001.mp4 