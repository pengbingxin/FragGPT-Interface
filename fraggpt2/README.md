## 安装：
pip install requirements.txt  
sudo apt-get install python3-dev

cd ./admet_predictor/datasets
python setup.py build_ext --inplace

## 
python generate_frag_pocket.py


## 训练
accelerate launch --config_file accelerate_config.yml train.py --config 

