echo "Generate Environments..."
python env.py \
    --dx 4 --du 2 \
    --inst_seed 42 \
    --sigma_square 0.1 \
    --save_dir "./envs"

python env.py \
    --dx 4 --du 2 \
    --inst_seed 43 \
    --sigma_square 0.1 \
    --save_dir "./envs"


echo "Running low_dim..."
python main.py --config configs/low_dim_warm.yaml
python main.py --config configs/one_simul.yaml