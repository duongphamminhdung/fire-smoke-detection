python test-on-collab.py \
        --name FIRE-ALARM \
        --model_path best.pt \
        --video_path test.mp4 \
        --max_age 7 \
        --min_hits 4 \
        --iou_threshold 0.3 \
        --skip-frame 5