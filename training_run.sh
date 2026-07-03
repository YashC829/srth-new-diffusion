#!/bin/bash

mamba activate srth-new
cd ~/surpass/srth-new

# has recovery data significantly improved our model's performance?
python src/srth_new/low_level_policy/train.py \
    dataloader=no_recovery \
    wandb.project=has_recovery_improved_performance \
    wandb.name=standard_data_no_recovery_no_kp

python src/srth_new/low_level_policy/train.py \
    dataloader=complete \
    wandb.project=has_recovery_improved_performance \
    wandb.name=with_recovery_no_kp

# what happens if we train a model on ALL of the data vs only one phase?
python src/srth_new/low_level_policy/train.py \
    dataloader=complete \
    dataloader.phases.unzipping=[4_hook_tissue,4_hook_tissue_recovery] \
    wandb.project=does_single_phase_training_improve_performance \
    wandb.name=hook_tissue_w_recovery_no_kp

# does keypoint model actually improve performance at all?
python src/srth_new/low_level_policy/train.py \
    dataloader=complete \
    policy=act_kp \
    dataloader.use_only_kp_annotated_data=True \
    dataloader.phases.unzipping=[4_hook_tissue,4_hook_tissue_recovery] \
    wandb.project=does_kp_model_improve \
    wandb.name=hook_tissue_w_recovery_with_kp

# does chunk size change performance?
python src/srth_new/low_level_policy/train.py \
    dataloader=complete \
    wandb.project=does_chunk_size_improve \
    wandb.name=complete_w_recovery_no_kp_chunk_size_10 \
    ll_future_chunk_size=10

python src/srth_new/low_level_policy/train.py \
    dataloader=complete \
    wandb.project=does_kp_model_improve \
    wandb.name=complete_w_recovery_no_kp_chunk_size_20 \
    ll_future_chunk_size=30

# verify that the corrected data works
python src/srth_new/low_level_policy/train.py \
    dataloader=test_data_correction \
    wandb.project=does_repaired_data_word \
    wandb.name=tissues_21_30