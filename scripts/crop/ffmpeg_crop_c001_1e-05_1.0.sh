ffmpeg -i /home/hongpeng/Desktop/research/AICity/Track3/train/S01/c001/vdo.avi -filter_complex "[0]split=5[s0][s1][s2][s3][s4]; [s0]crop=1536:448:384:128[s0]; [s1]crop=704:256:384:576[s1]; [s2]crop=512:256:1408:576[s2]; [s3]crop=384:320:0:192[s3]; [s4]crop=192:64:192:128[s4]" -map [s0] -c:v libx264 videos/S01/1e-05_1.0/tmp/c001_00.mp4 -map [s1] -c:v libx264 videos/S01/1e-05_1.0/tmp/c001_01.mp4 -map [s2] -c:v libx264 videos/S01/1e-05_1.0/tmp/c001_02.mp4 -map [s3] -c:v libx264 videos/S01/1e-05_1.0/tmp/c001_03.mp4 -map [s4] -c:v libx264 videos/S01/1e-05_1.0/tmp/c001_04.mp4 