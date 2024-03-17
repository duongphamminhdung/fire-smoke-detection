pip install -r requirements.txt

wget https://drive.google.com/uc?id=1JEnQRmeySnz2fUMdnB0SBVPuS9BderLk -O test.mp4

wget https://drive.google.com/uc?id=1WlV0km_w0DjAbDCrCBYvZc211S5tMKIZ -O /root/fire-smoke-detection/test.jpg
wget https://drive.google.com/uc?id=1qfDNa-Ezedfbe7ZLDx-m33QBMg1mllHD -O /root/fire-smoke-detection/best.pt #new model

# # train data
# pip install gdown
# mv cookies.txt ~/.cache/gdown
# gdown https://drive.google.com/uc?id=1N0vVJJXhgVGgKMpFJxpuPD_n-u-Rhz8a -O data.tar
# tar -xvf data.tar
# wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt