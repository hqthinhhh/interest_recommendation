
// từ tuổi -> sở thích
// từ giới tính -> sở thích
// kết hợp tuổi + giới tính -> sở thích

nguồn dataset: https://www.kaggle.com/datasets/arindamsahoo/social-media-users
folder dataset là data ban đầu để tạo rule

results là kết quả rule được tạo
step chạy test:

pip install -r requirements.txt
(phải cài db postgres)
uvicorn api:app --reload