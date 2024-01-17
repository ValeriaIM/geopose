PYTHONPATH=. python utilities/cythonize_invert_flow.py build_ext --inplace

DATA="/home/s0105/_scratch2/data/rgb-lab.png"
GPUS="1"

PYTHONPATH=.    python3 -u -m torch.distributed.run \
--nproc_per_node=$GPUS \
main.py \
--world_size $GPUS \
--predict \
--predictions-dir=submission \
--dataset-dir=$DATA \
--model-path weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_5_r2 weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_6_r2 \
--batch-size=1 \
--gpus=1 \
--unit="cm" \
--downsample=1 \
--test-sub-dir="" \
--convert-predictions-to-cm-and-compress=True \
--rgb-suffix="tif"