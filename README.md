# CMMLoc_MNCLv3

## Coarse stage 
1. python -m training.coarse --batch_size 128 --coarse_embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --use_features "class" "color" "position" "num" --no_pc_augment --fixed_embedding --epochs 20 --learning_rate 0.0005 --lr_scheduler step --lr_step 7 --lr_gamma 0.4 --temperature 0.1 --ranking_loss contrastive --hungging_model t5-large --text_max_length 128 --lambda_mncl 0.3 --mncl_proj_dim 256 --use_trainable_reranker --lambda_reranker 0.5 --reranker_hidden_dim 32 --trainable_rerank_topn 50 --cpus 0 --pointnet_path ./checkpoints/pointnet_acc0.86_lr1_p256.pth --folder_name cmmlocv3_batch128_epoch20
2. python -m training.coarse --batch_size 64 --coarse_embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --use_features "class" "color" "position" "num" --no_pc_augment --fixed_embedding --epochs 20 --learning_rate 0.0005 --lr_scheduler step --lr_step 7 --lr_gamma 0.4 --temperature 0.1 --ranking_loss contrastive --hungging_model t5-large --text_max_length 128 --lambda_mncl 0.3 --mncl_proj_dim 256 --use_trainable_reranker --lambda_reranker 0.5 --reranker_hidden_dim 32 --trainable_rerank_topn 50 --cpus 0 --pointnet_path ./checkpoints/pointnet_acc0.86_lr1_p256.pth --folder_name cmmlocv3_batch64_epoch20


