

python inference_ccsr.py \
--input preset/test_datasets \
--config configs/model/ccsr_stage2.yaml \
--ckpt weights/real-world_ccsr.ckpt \
--steps 45 \
--sr_scale 4 \
--t_max 0.6667 \
--t_min 0.3333 \
--color_fix_type adain \
--output experiments/test \
--device cuda \
--repeat_times 1 
