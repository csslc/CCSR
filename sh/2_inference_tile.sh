

python inference_ccsr_tile.py \
--input preset/test_datasets \
--config configs/model/ccsr_stage2.yaml \
--ckpt /home/notebook/data/group/SunLingchen/code/CCSR/CCSR_weights/real-world_ccsr.ckpt \
--steps 45 \
--sr_scale 4 \
--t_max 0.6667 \
--t_min 0.3333 \
--tile_diffusion \
--tile_diffusion_size 512 \
--tile_diffusion_stride 256 \
--tile_vae \
--vae_decoder_tile_size 224 \
--vae_encoder_tile_size 1024 \
--color_fix_type adain \
--output experiments/test \
--device cuda \
--repeat_times 1