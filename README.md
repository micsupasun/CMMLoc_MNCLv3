# CMMLoc_MNCLv3

## Coarse stage 
1. python -m training.coarse --batch_size 128 --coarse_embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --use_features "class" "color" "position" "num" --no_pc_augment --fixed_embedding --epochs 20 --learning_rate 0.0005 --lr_scheduler step --lr_step 7 --lr_gamma 0.4 --temperature 0.1 --ranking_loss contrastive --hungging_model t5-large --text_max_length 128 --lambda_mncl 0.3 --mncl_proj_dim 256 --use_trainable_reranker --lambda_reranker 0.5 --reranker_hidden_dim 32 --trainable_rerank_topn 50 --cpus 0 --pointnet_path ./checkpoints/pointnet_acc0.86_lr1_p256.pth --folder_name cmmlocv3_batch128_epoch20
2. python -m training.coarse --batch_size 64 --coarse_embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --use_features "class" "color" "position" "num" --no_pc_augment --fixed_embedding --epochs 20 --learning_rate 0.0005 --lr_scheduler step --lr_step 7 --lr_gamma 0.4 --temperature 0.1 --ranking_loss contrastive --hungging_model t5-large --text_max_length 128 --lambda_mncl 0.3 --mncl_proj_dim 256 --use_trainable_reranker --lambda_reranker 0.5 --reranker_hidden_dim 32 --trainable_rerank_topn 50 --cpus 0 --pointnet_path ./checkpoints/pointnet_acc0.86_lr1_p256.pth --folder_name cmmlocv3_batch64_epoch20





## Evaluation Ash

- If you need Trained reranker coarse + resumed fine epoch 44 in the evaluation command(method 1)
- cmmlocpp_mncl_rerank_v1
- cmmlocpp_mncl_rerank_v1_fine_resume
(Coarse+Fine) Evaluation Validation command:

python -m evaluation.pipeline --base_path "E:\Github storage\CMMLocPP\data\k360_30-10_scG_pd10_pc4_spY_all" --use_features class color position num --no_pc_augment --no_pc_augment_fine --hungging_model t5-large --fixed_embedding --text_max_length 96 --use_model_reranker --rerank_topn 50 --path_coarse "/share/nas/cs-nas/zh932237/CMMLoc_MNCLv3/checkpoints/k360_30-10_scG_pd10_pc4_spY_all/ash_file/cmmlocpp_mncl_rerank_v1/coarse_contN_epoch3_acc0.796_ecl0_eco0_p256_npa1_loss-contrastive_f-class-color-position-num.pth" --path_fine "/share/nas/cs-nas/zh932237/CMMLoc_MNCLv3/checkpoints/k360_30-10_scG_pd10_pc4_spY_all/ash_file/cmmlocpp_mncl_rerank_v1_fine_resume/fine_contY_epoch44_offset0.108_lr0.0003_obj-6-16_ecl0_eco0_p256_npa1_f-class-color-position-num.pth"


(Coarse+Fine) Evaluation Test command:

python -m evaluation.pipeline --base_path "E:\Github storage\CMMLocPP\data\k360_30-10_scG_pd10_pc4_spY_all" --use_test_set --use_features class color position num --no_pc_augment --no_pc_augment_fine --hungging_model t5-large --fixed_embedding --text_max_length 96 --use_model_reranker --rerank_topn 50 --path_coarse "/share/nas/cs-nas/zh932237/CMMLoc_MNCLv3/checkpoints/k360_30-10_scG_pd10_pc4_spY_all/ash_file/cmmlocpp_mncl_rerank_v1/coarse_contN_epoch3_acc0.796_ecl0_eco0_p256_npa1_loss-contrastive_f-class-color-position-num.pth" --path_fine "/share/nas/cs-nas/zh932237/CMMLoc_MNCLv3/checkpoints/k360_30-10_scG_pd10_pc4_spY_all/ash_file/cmmlocpp_mncl_rerank_v1_fine_resume/fine_contY_epoch44_offset0.108_lr0.0003_obj-6-16_ecl0_eco0_p256_npa1_f-class-color-position-num.pth"

(Coarse Only) Evaluation Validation Command

python -m evaluation.pipeline --base_path "E:\Github storage\CMMLocPP\data\k360_30-10_scG_pd10_pc4_spY_all" --use_features class color position num --no_pc_augment --no_pc_augment_fine --hungging_model t5-large --fixed_embedding --text_max_length 96 --coarse_only --use_model_reranker --rerank_topn 50 --path_coarse "/share/nas/cs-nas/zh932237/CMMLoc_MNCLv3/checkpoints/k360_30-10_scG_pd10_pc4_spY_all/ash_file/cmmlocpp_mncl_rerank_v1/coarse_contN_epoch3_acc0.796_ecl0_eco0_p256_npa1_loss-contrastive_f-class-color-position-num.pth"

(Coarse Only) Evaluation Test Command

python -m evaluation.pipeline --base_path "E:\Github storage\CMMLocPP\data\k360_30-10_scG_pd10_pc4_spY_all" --use_test_set --use_features class color position num --no_pc_augment --no_pc_augment_fine --hungging_model t5-large --fixed_embedding --text_max_length 96 --coarse_only --use_model_reranker --rerank_topn 50 --path_coarse "/share/nas/cs-nas/zh932237/CMMLoc_MNCLv3/checkpoints/k360_30-10_scG_pd10_pc4_spY_all/ash_file/cmmlocpp_mncl_rerank_v1/coarse_contN_epoch3_acc0.796_ecl0_eco0_p256_npa1_loss-contrastive_f-class-color-position-num.pth"


------------------------------------------------
- If you need Trained reranker coarse + old fine then kindly load these files in evaluation command(method 2)
- cmmlocpp_mncl_rerank_v1
- cmmlocpp_mncl_v2_fine
(Coarse+Fine) Evaluation Validation command:

python -m evaluation.pipeline --base_path "E:\Github storage\CMMLocPP\data\k360_30-10_scG_pd10_pc4_spY_all" --use_features class color position num --no_pc_augment --no_pc_augment_fine --hungging_model t5-large --fixed_embedding --text_max_length 96 --use_model_reranker --rerank_topn 50 --path_coarse "/share/nas/cs-nas/zh932237/CMMLoc_MNCLv3/checkpoints/k360_30-10_scG_pd10_pc4_spY_all/ash_file/cmmlocpp_mncl_rerank_v1/coarse_contN_epoch3_acc0.796_ecl0_eco0_p256_npa1_loss-contrastive_f-class-color-position-num.pth" --path_fine "/share/nas/cs-nas/zh932237/CMMLoc_MNCLv3/checkpoints/k360_30-10_scG_pd10_pc4_spY_all/ash_file/cmmlocpp_mncl_v2_fine/fine_contN_epoch41_offset0.103_lr0.0003_obj-6-16_ecl0_eco0_p256_npa1_f-class-color-position-num.pth"


(Coarse+Fine) Evaluation Test command:

python -m evaluation.pipeline --base_path "E:\Github storage\CMMLocPP\data\k360_30-10_scG_pd10_pc4_spY_all" --use_test_set --use_features class color position num --no_pc_augment --no_pc_augment_fine --hungging_model t5-large --fixed_embedding --text_max_length 96 --use_model_reranker --rerank_topn 50 --path_coarse "/share/nas/cs-nas/zh932237/CMMLoc_MNCLv3/checkpoints/k360_30-10_scG_pd10_pc4_spY_all/ash_file/cmmlocpp_mncl_rerank_v1/coarse_contN_epoch3_acc0.796_ecl0_eco0_p256_npa1_loss-contrastive_f-class-color-position-num.pth" --path_fine "/share/nas/cs-nas/zh932237/CMMLoc_MNCLv3/checkpoints/k360_30-10_scG_pd10_pc4_spY_all/ash_file/cmmlocpp_mncl_v2_fine/fine_contN_epoch41_offset0.103_lr0.0003_obj-6-16_ecl0_eco0_p256_npa1_f-class-color-position-num.pth"

(Coarse Only) Evaluation Validation Command

python -m evaluation.pipeline --base_path "E:\Github storage\CMMLocPP\data\k360_30-10_scG_pd10_pc4_spY_all" --use_features class color position num --no_pc_augment --no_pc_augment_fine --hungging_model t5-large --fixed_embedding --text_max_length 96 --coarse_only --use_model_reranker --rerank_topn 50 --path_coarse "/share/nas/cs-nas/zh932237/CMMLoc_MNCLv3/checkpoints/k360_30-10_scG_pd10_pc4_spY_all/ash_file/cmmlocpp_mncl_rerank_v1/coarse_contN_epoch3_acc0.796_ecl0_eco0_p256_npa1_loss-contrastive_f-class-color-position-num.pth"

(Coarse Only) Evaluation Test Command

python -m evaluation.pipeline --base_path "E:\Github storage\CMMLocPP\data\k360_30-10_scG_pd10_pc4_spY_all" --use_test_set --use_features class color position num --no_pc_augment --no_pc_augment_fine --hungging_model t5-large --fixed_embedding --text_max_length 96 --coarse_only --use_model_reranker --rerank_topn 50 --path_coarse "/share/nas/cs-nas/zh932237/CMMLoc_MNCLv3/checkpoints/k360_30-10_scG_pd10_pc4_spY_all/ash_file/cmmlocpp_mncl_rerank_v1/coarse_contN_epoch3_acc0.796_ecl0_eco0_p256_npa1_loss-contrastive_f-class-color-position-num.pth"




