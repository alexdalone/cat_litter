#Run with the following command:
cd ~/hailo-apps-infra
source setup_env.sh
cd ~/cat_litter_monitor


python hailo_streamer.py \
  --hef /usr/local/hailo/resources/models/hailo8/yolov8m.hef \
  --input /dev/video0 \
  --use-frame \
  --disable-sync \
  --batch-size 1 \
  --frame-rate 120
