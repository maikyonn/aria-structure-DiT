wget -O aria-midi-v1-pruned-ext.tar.gz "https://huggingface.co/datasets/loubb/aria-midi/resolve/main/aria-midi-v1-pruned-ext.tar.gz?download=true"
tar -I pigz -xf aria-midi-v1-pruned-ext.tar.gz



conda create -n aria-dit python=3.11

pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu128
pip install ninja
pip install packaging

git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention 
export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32 # parallel compiling (Optional)
python setup.py install  # or pip install -e .

git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/csrc/layer_norm
python setup.py install
cd ../rotary
python setup.py install
cd ../xentropy
python setup.py install

cd ../../..
rm -rf flash-attention
rm -rf SageAttention

pip install mido 
pip install wandb
pip install lightning
pip install pytorch-lightning
pip install tqdm
pip install einops

python train.py \
  --train_pkl cache/dataset_paths_real_data_limitNone_c79653f4_train.pkl \
  --val_pkl cache/dataset_paths_real_data_limitNone_c79653f4_val.pkl \
  --objective pure \
  --model 336 \
  --max_epochs 20 \
  --devices 4 --batch_size 16 --num_workers 8 \
  --accumulate_grad_batches 8 \
  --lr 3e-4 --weight_decay 1e-2 \
  --decay_lr --warmup_iters 2000 --min_lr 1e-5 \
  --mask_prob_min 0.03 --mask_prob_max 0.30 --mask_schedule linear \
  --use_swa --swa_lr 1e-4 --swa_epoch_start 15 \
  --wandb_project scaling

