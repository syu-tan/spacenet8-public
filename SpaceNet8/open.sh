# open beseline 
## env
docker build -t sn8/baseline:1.0 -f docker/Dockerfile  .
docker build -t sn8/baseline:1.1 -f docker/Dockerfile  .
docker build -t sn8/baseline:1.2 -f docker/Dockerfile  .
docker build -t sn8/baseline:1.3 -f docker/Dockerfile  .
docker run -it --shm-size=64g --gpus all -v /media/syu/c983cccd-1cc7-4ce4-b206-7eb1a2cc5c94/topcoder/spacenet8/:/workdir/ --rm sn8/baseline:1.0 bash
docker run -it --shm-size=64g --gpus all -v /media/syu/c983cccd-1cc7-4ce4-b206-7eb1a2cc5c94/topcoder/spacenet8/:/workdir/ --rm sn8/baseline:1.2 bash
docker run -it --shm-size=64g --gpus all -v /media/syu/c983cccd-1cc7-4ce4-b206-7eb1a2cc5c94/topcoder/spacenet8/:/workdir/ --rm sn8/baseline:1.3 bash

docker run -it --shm-size=64g --gpus all -v /media/syu/c983cccd-1cc7-4ce4-b206-7eb1a2cc5c94/topcoder/spacenet8/:/workdir/ --rm sn8/baseline:sub bash
docker run -it --shm-size=64g --gpus all -v $PWD:/workdir/ --rm sn8/baseline:sub bash

# CPU
docker run -it --shm-size=64g  -v /Users/syu/src/git/spacenet8/:/workdir/ --rm sn8/baseline:1.3 bash

# windows
docker run -it --shm-size=64g --gpus all -v C:\Users\spcsf\src\spacenet8\:/workdir/ --rm sn8/baseline:1.2 bash

# GPU check
python -c "import torch; print(torch.cuda.is_available())"

## prepare
cd /workdir/SpaceNet8/
python baseline/data_prep/geojson_prep.py
python baseline/data_prep/create_masks.py
python baseline/data_prep/generate_train_val_test_csvs.py

# train
## feature
python baseline/train_foundation_features.py \
    --train_csv /workdir/data/folds/open_baseline_seed-417_val-0.2_train.csv \
    --val_csv /workdir/data/folds/open_baseline_seed-417_val-0.2_val.csv \
     --gpu 0 \
    --save_dir ../train/output/open_baseline/ --model_name resnet34 --lr 0.0002 --batch_size 4 --n_epochs 50

# inference
python baseline/foundation_eval.py --model_path ../train/output/open_baseline/resnet34_lr2.00e-04_bs4_16-07-2022-12-15/best_model.pth \
    --in_csv ../data/folds/open_baseline_seed-417_val-0.2_val.csv --gpu 0 --model_name resnet34 \
    --save_preds_dir ../train/output/open_baseline/resnet34_lr2.00e-04_bs4_16-07-2022-12-15/val/
python baseline/foundation_inf.py --model_path ../train/output/open_baseline/resnet34_lr2.00e-04_bs4_16-07-2022-12-15/best_model.pth \
    --in_csv ../data/Louisiana-West_Test_Public/open_baseline_test.csv \
     --gpu 0 --model_name resnet34 \
    --save_preds_dir ../train/output/open_baseline/resnet34_lr2.00e-04_bs4_16-07-2022-12-15/test/
# visualize
python baseline/foundation_eval.py --model_path ../train/output/open_baseline/resnet34_lr2.00e-04_bs4_16-07-2022-12-15/best_model.pth \
    --in_csv ../data/folds/open_baseline_seed-417_val-0.2_val.csv \
    --save_fig_dir ../train/output/open_baseline/resnet34_lr2.00e-04_bs4_16-07-2022-12-15/png/ --gpu 0 --model_name resnet34 


## flood
python baseline/train_flood.py --train_csv /workdir/data/folds/open_baseline_seed-417_val-0.2_train.csv \
    --val_csv /workdir/data/folds/open_baseline_seed-417_val-0.2_val.csv \
    --save_dir ../train/output/open_baseline_flood/ --model_name resnet34 \
    --lr 0.0001 --batch_size 2 --n_epochs 50 --gpu 0

# inference
python baseline/flood_eval.py \
    --model_path ../train/output/open_baseline_flood/resnet34_siamese_lr1.00e-04_bs2_16-07-2022-14-19/best_model.pth \
    --in_csv ../data/folds/open_baseline_seed-417_val-0.2_val.csv \
    --save_preds_dir ../train/output/open_baseline_flood/resnet34_siamese_lr1.00e-04_bs2_16-07-2022-14-19/val/ \
    --gpu 0 --model_name resnet34_siamese
python baseline/flood_inf.py \
    --model_path ../train/output/open_baseline_flood/resnet34_siamese_lr1.00e-04_bs2_16-07-2022-14-19/best_model.pth \
    --in_csv ../data/Louisiana-West_Test_Public/open_baseline_test.csv \
    --save_preds_dir ../train/output/open_baseline_flood/resnet34_siamese_lr1.00e-04_bs2_16-07-2022-14-19/test/ \
    --gpu 0 --model_name resnet34_siamese
# visualize
python baseline/flood_eval.py \
    --model_path ../train/output/open_baseline_flood/resnet34_siamese_lr1.00e-04_bs2_16-07-2022-14-19/best_model.pth \
    --in_csv ../data/folds/open_baseline_seed-417_val-0.2_val.csv \
    --save_fig_dir ../train/output/open_baseline_flood/resnet34_siamese_lr1.00e-04_bs2_16-07-2022-14-19/png/ \
    --gpu 0 --model_name resnet34_siamese
python baseline/flood_inf.py \
    --model_path ../train/output/open_baseline_flood/resnet34_siamese_lr1.00e-04_bs2_16-07-2022-14-19/best_model.pth \
    --in_csv ../data/Louisiana-West_Test_Public/open_baseline_test.csv \
    --save_fig_dir ../train/output/open_baseline_flood/resnet34_siamese_lr1.00e-04_bs2_16-07-2022-14-19/test/png/ \
    --gpu 0 --model_name resnet34_siamese

## postprocess foundmental
./baseline/postprocessing/roads/road_post.sh
./baseline/postprocessing/roads/road_post_test.sh

## postprocess flood
python baseline/postprocessing/buildings/building_postprocessing.py \
    --foundation_pred_dir ../train/output/open_baseline/resnet34_lr2.00e-04_bs4_16-07-2022-12-15/val/ \
    --flood_pred_dir ../train/output/open_baseline_flood/resnet34_siamese_lr1.00e-04_bs2_16-07-2022-14-19/val/ \
    --out_submission_csv ../train/output/open_baseline_flood/resnet34_siamese_lr1.00e-04_bs2_16-07-2022-14-19/val/building_submission.csv \
    --out_shapefile_dir ../train/output/open_baseline_flood/resnet34_siamese_lr1.00e-04_bs2_16-07-2022-14-19/val/ \
    --square_size 5 --simplify_tolerance 0.75 --min_area 5 --percent_positive 0.5
python baseline/postprocessing/buildings/building_postprocessing.py \
    --foundation_pred_dir ../train/output/open_baseline/resnet34_lr2.00e-04_bs4_16-07-2022-12-15/test/ \
    --flood_pred_dir ../train/output/open_baseline_flood/resnet34_siamese_lr1.00e-04_bs2_16-07-2022-14-19/test/ \
    --out_submission_csv ../train/output/open_baseline_flood/resnet34_siamese_lr1.00e-04_bs2_16-07-2022-14-19/test/building_submission.csv \
    --out_shapefile_dir ../train/output/open_baseline_flood/resnet34_siamese_lr1.00e-04_bs2_16-07-2022-14-19/test/shapefile/ \
    --square_size 5 --simplify_tolerance 0.75 --min_area 5 --percent_positive 0.5

# my baseline
python baseline/postprocessing/buildings/building_postprocessing.py \
    --foundation_pred_dir ../train/output/foundation/A_V1_IMG1280_eff0-dec512_clean-v1_fl-tr-hsv-blbr/A_fold-4/test_fnd/ \
    --flood_pred_dir ../train/output/flood/3090_V1_IMG1280_flip-trans/3090_fold-4/test_fld/ \
    --out_submission_csv ../train/output/foundation/A_V1_IMG1280_eff0-dec512_clean-v1_fl-tr-hsv-blbr/A_fold-4/test_fnd/solution_building.csv \
    --out_shapefile_dir ../train/output/foundation/A_V1_IMG1280_eff0-dec512_clean-v1_fl-tr-hsv-blbr/A_fold-4/test_fnd/shapefile/ \
    --square_size 5 --simplify_tolerance 0.5 --min_area 5 --percent_positive 0.5

# 70.96
python baseline/postprocessing/buildings/building_postprocessing.py \
    --foundation_pred_dir ../train/output/foundation/3090_V2_timmu_IMG1280_fl-tr-hsv/3090_fold-4/test_fnd/ \
    --flood_pred_dir ../train/output/flood/3090_V1_IMG1280_flip-trans/3090_fold-4/test_fld/ \
    --out_submission_csv ../train/output/foundation/3090_V2_timmu_IMG1280_fl-tr-hsv/3090_fold-4/test_fnd/solution_building.csv \
    --out_shapefile_dir ../train/output/foundation/3090_V2_timmu_IMG1280_fl-tr-hsv/3090_fold-4/test_fnd/shapefile/ \
    --square_size 5 --simplify_tolerance 0.5 --min_area 5 --percent_positive 0.5

# 70.62
python baseline/postprocessing/buildings/building_postprocessing.py \
    --foundation_pred_dir ../train/output/foundation/A_V5_clean-v2__timmu_IMG1312_hrnet_w30_fl-tr-hsv-blbr_b8/A_fold-4/test_fnd/ \
    --flood_pred_dir ../train/output/flood/A_V5_clean-v3_timmu_IMG1312_hrnet_w30_fl-tr_b8/A_fold-4/test_fld/ \
    --out_submission_csv ../train/output/foundation/A_V5_clean-v2__timmu_IMG1312_hrnet_w30_fl-tr-hsv-blbr_b8/A_fold-4/test_fnd/solution_building.csv \
    --out_shapefile_dir ../train/output/foundation/A_V5_clean-v2__timmu_IMG1312_hrnet_w30_fl-tr-hsv-blbr_b8/A_fold-4/test_fnd/shapefile/ \
    --square_size 5 --simplify_tolerance 0.5 --min_area 5 --percent_positive 0.5

# 72.24
python baseline/postprocessing/buildings/building_postprocessing.py \
    --foundation_pred_dir ../train/output/foundation/A_V5_clean-v2__timmu_IMG1312_hrnet_w30_fl-tr-hsv-blbr_b8/A_fold-4/test_fnd/ \
    --flood_pred_dir ../train/output/flood/3090_V1_IMG1280_flip-trans/3090_fold-4/test_fld/ \
    --out_submission_csv ../train/output/foundation/A_V5_clean-v2__timmu_IMG1312_hrnet_w30_fl-tr-hsv-blbr_b8/A_fold-4/test_fnd/solution_building.csv \
    --out_shapefile_dir ../train/output/foundation/A_V5_clean-v2__timmu_IMG1312_hrnet_w30_fl-tr-hsv-blbr_b8/A_fold-4/test_fnd/shapefile/ \
    --square_size 5 --simplify_tolerance 0.5 --min_area 5 --percent_positive 0.5

# 71.38
python baseline/postprocessing/buildings/building_postprocessing.py \
    --foundation_pred_dir ../train/output/foundation/A_V5_POST_clean-v2__timmu_IMG1312_hrnet_w30_fl-tr-blbr_b8/A_fold-0/test_fnd/ \
    --flood_pred_dir ../train/output/flood/3090_V1_IMG1280_flip-trans/3090_fold-4/test_fld/ \
    --out_submission_csv ../train/output/foundation/A_V5_POST_clean-v2__timmu_IMG1312_hrnet_w30_fl-tr-blbr_b8/A_fold-0/test_fnd/solution_building.csv \
    --out_shapefile_dir ../train/output/foundation/A_V5_POST_clean-v2__timmu_IMG1312_hrnet_w30_fl-tr-blbr_b8/A_fold-0/test_fnd/shapefile/ \
    --square_size 5 --simplify_tolerance 0.5 --min_area 5 --percent_positive 0.5

# 70.14 -> png がはっきりしていない
python baseline/postprocessing/buildings/building_postprocessing.py \
    --foundation_pred_dir ../train/output/foundation/3090_V5_fulltrain__unet_IMG1280_resnet34-dep4_fl-tr-hsv-blbr_b8_/3090_fold-0/test_fnd/ \
    --flood_pred_dir ../train/output/flood/3090_V1_IMG1280_flip-trans/3090_fold-4/test_fld/ \
    --out_submission_csv ../train/output/foundation/3090_V5_fulltrain__unet_IMG1280_resnet34-dep4_fl-tr-hsv-blbr_b8_/3090_fold-0/test_fnd/solution_building.csv \
    --out_shapefile_dir ../train/output/foundation/3090_V5_fulltrain__unet_IMG1280_resnet34-dep4_fl-tr-hsv-blbr_b8_/3090_fold-0/test_fnd/shapefile/ \
    --square_size 5 --simplify_tolerance 0.5 --min_area 5 --percent_positive 0.5

# 71.72 hanshold
python baseline/postprocessing/buildings/building_postprocessing.py \
    --foundation_pred_dir ../train/output/foundation/A_V5_clean-v2_timmu_IMG1312_hrnet_w30_fl-tr_b8/A_fold-0/test_fnd/ \
    --flood_pred_dir ../train/output/flood/3090_V1_IMG1280_flip-trans/3090_fold-4/test_fld/ \
    --out_submission_csv ../train/output/foundation/A_V5_clean-v2_timmu_IMG1312_hrnet_w30_fl-tr_b8/A_fold-0/test_fnd/solution_building.csv \
    --out_shapefile_dir ../train/output/foundation/A_V5_clean-v2_timmu_IMG1312_hrnet_w30_fl-tr_b8/A_fold-0/test_fnd/shapefile/ \
    --square_size 5 --simplify_tolerance 0.5 --min_area 5 --percent_positive 0.5

# 72.24 ** re-inference -> 72.19 ???
python baseline/postprocessing/buildings/building_postprocessing.py \
    --foundation_pred_dir ../train/output/foundation/A_V5_clean-v2__timmu_IMG1312_hrnet_w30_fl-tr-hsv-blbr_b8/A_fold-4/test_fnd/ \
    --flood_pred_dir ../train/output/flood/3090_V1_IMG1280_flip-trans/3090_fold-4/test_fld/ \
    --out_submission_csv ../train/output/foundation/A_V5_clean-v2__timmu_IMG1312_hrnet_w30_fl-tr-hsv-blbr_b8/A_fold-4/test_fnd/solution_building.csv \
    --out_shapefile_dir ../train/output/foundation/A_V5_clean-v2__timmu_IMG1312_hrnet_w30_fl-tr-hsv-blbr_b8/A_fold-4/test_fnd/shapefile/ \
    --square_size 5 --simplify_tolerance 0.5 --min_area 5 --percent_positive 0.5

    train/output/flood/3090_V5__timmu_IMG1312_hrnet_w30_fl-tr/3090_fold-4/test_fld

# 72.39
python baseline/postprocessing/buildings/building_postprocessing.py \
    --foundation_pred_dir ../train/output/foundation/A_V5_clean-v2__timmu_IMG1312_hrnet_w30_fl-tr-hsv-blbr_b8/A_fold-4/test_fnd/ \
    --flood_pred_dir ../train/output/flood/3090_V5__timmu_IMG1312_hrnet_w30_fl-tr/3090_fold-4/test_fld/ \
    --out_submission_csv ../train/output/foundation/A_V5_clean-v2__timmu_IMG1312_hrnet_w30_fl-tr-hsv-blbr_b8/A_fold-4/test_fnd/solution_building.csv \
    --out_shapefile_dir ../train/output/foundation/A_V5_clean-v2__timmu_IMG1312_hrnet_w30_fl-tr-hsv-blbr_b8/A_fold-4/test_fnd/shapefile/ \
    --square_size 5 --simplify_tolerance 0.5 --min_area 5 --percent_positive 0.5

# 72.15
python baseline/postprocessing/buildings/building_postprocessing.py \
    --foundation_pred_dir ../train/output/foundation/3090_V6_POST_clean-v2__timmu_IMG1280_hrnet_w30-dec384-4_fl-tr_b4/3090_fold-4/test_fnd/ \
    --flood_pred_dir ../train/output/flood/3090_V5__timmu_IMG1312_hrnet_w30_fl-tr/3090_fold-4/test_fld/ \
    --out_submission_csv ../train/output/foundation/3090_V6_POST_clean-v2__timmu_IMG1280_hrnet_w30-dec384-4_fl-tr_b4/3090_fold-4/test_fnd/solution_building.csv \
    --out_shapefile_dir ../train/output/foundation/3090_V6_POST_clean-v2__timmu_IMG1280_hrnet_w30-dec384-4_fl-tr_b4/3090_fold-4/test_fnd/shapefile/ \
    --square_size 5 --simplify_tolerance 0.5 --min_area 5 --percent_positive 0.5

# 72.39 ** 72.30
python baseline/postprocessing/buildings/building_postprocessing.py \
    --foundation_pred_dir ../train/output/foundation/A_V5_clean-v2__timmu_IMG1312_hrnet_w30_fl-tr-hsv-blbr_b8/A_fold-4/test_fnd/ \
    --flood_pred_dir ../train/output/flood/3090_V5__timmu_IMG1312_hrnet_w30_fl-tr/3090_fold-4/test_fld/ \
    --out_submission_csv ../train/output/foundation/A_V5_clean-v2__timmu_IMG1312_hrnet_w30_fl-tr-hsv-blbr_b8/A_fold-4/test_fnd/solution_building.csv \
    --out_shapefile_dir ../train/output/foundation/A_V5_clean-v2__timmu_IMG1312_hrnet_w30_fl-tr-hsv-blbr_b8/A_fold-4/test_fnd/shapefile/ \
    --square_size 5 --simplify_tolerance 0.5 --min_area 5 --percent_positive 0.4

# 72.39 ** 72.38
python baseline/postprocessing/buildings/building_postprocessing.py \
    --foundation_pred_dir ../train/output/foundation/A_V5_clean-v2__timmu_IMG1312_hrnet_w30_fl-tr-hsv-blbr_b8/A_fold-4/test_fnd/ \
    --flood_pred_dir ../train/output/flood/3090_V5__timmu_IMG1312_hrnet_w30_fl-tr/3090_fold-4/test_fld/ \
    --out_submission_csv ../train/output/foundation/A_V5_clean-v2__timmu_IMG1312_hrnet_w30_fl-tr-hsv-blbr_b8/A_fold-4/test_fnd/solution_building.csv \
    --out_shapefile_dir ../train/output/foundation/A_V5_clean-v2__timmu_IMG1312_hrnet_w30_fl-tr-hsv-blbr_b8/A_fold-4/test_fnd/shapefile/ \
    --square_size 5 --simplify_tolerance 0.5 --min_area 5 --percent_positive 0.6


# 73.21
python baseline/postprocessing/buildings/building_postprocessing.py \
    --foundation_pred_dir ../train/output/foundation/A_V5_clean-v2__timmu_IMG1312_hrnet_w30_fl-tr-hsv-blbr_b8/A_fold-4/test_fnd/ \
    --flood_pred_dir ../train/output/flood/3090_V5_clean-v3__timmu_IMG1312_hrnet_w18_fl-tr_b4/3090_fold-4/test_fld/ \
    --out_submission_csv ../train/output/foundation/A_V5_clean-v2__timmu_IMG1312_hrnet_w30_fl-tr-hsv-blbr_b8/A_fold-4/test_fnd/solution_building.csv \
    --out_shapefile_dir ../train/output/foundation/A_V5_clean-v2__timmu_IMG1312_hrnet_w30_fl-tr-hsv-blbr_b8/A_fold-4/test_fnd/shapefile/ \
    --square_size 5 --simplify_tolerance 0.5 --min_area 5 --percent_positive 0.5

# 73.27
python baseline/postprocessing/buildings/building_postprocessing.py \
    --foundation_pred_dir ../train/output/foundation/A_V6_clean-v2__timmu_IMG1312_hrnet_w48-dec256_fl-tr_b4/A_fold-4/test_fnd/ \
    --flood_pred_dir ../train/output/flood/3090_V5_clean-v3__timmu_IMG1312_hrnet_w18_fl-tr_b4/3090_fold-4/test_fld/ \
    --out_submission_csv ../train/output/foundation/A_V6_clean-v2__timmu_IMG1312_hrnet_w48-dec256_fl-tr_b4/A_fold-4/test_fnd/solution_building.csv \
    --out_shapefile_dir ../train/output/foundation/A_V6_clean-v2__timmu_IMG1312_hrnet_w48-dec256_fl-tr_b4/A_fold-4/test_fnd/shapefile/ \
    --square_size 5 --simplify_tolerance 0.5 --min_area 5 --percent_positive 0.5

# 72.72
python baseline/postprocessing/buildings/building_postprocessing.py \
    --foundation_pred_dir ../train/output/foundation/3090_V6_clean-v2__timmu_IMG1312_hrnet_w64-dec256-4_fl-tr_b2/3090_fold-0/test_fnd/ \
    --flood_pred_dir ../train/output/flood/3090_V5_clean-v3__timmu_IMG1312_hrnet_w18_fl-tr_b4/3090_fold-4/test_fld/ \
    --out_submission_csv ../train/output/foundation/3090_V6_clean-v2__timmu_IMG1312_hrnet_w64-dec256-4_fl-tr_b2/3090_fold-0/test_fnd/solution_building.csv \
    --out_shapefile_dir ../train/output/foundation/3090_V6_clean-v2__timmu_IMG1312_hrnet_w64-dec256-4_fl-tr_b2/3090_fold-0/test_fnd/shapefile/ \
    --square_size 5 --simplify_tolerance 0.5 --min_area 5 --percent_positive 0.5

# 73.27 flood th 0.3 > 73.26
python baseline/postprocessing/buildings/building_postprocessing.py \
    --foundation_pred_dir ../train/output/foundation/A_V6_clean-v2__timmu_IMG1312_hrnet_w48-dec256_fl-tr_b4/A_fold-4/test_fnd/ \
    --flood_pred_dir ../train/output/flood/3090_V5_clean-v3__timmu_IMG1312_hrnet_w18_fl-tr_b4/3090_fold-4/test_fld/ \
    --out_submission_csv ../train/output/foundation/A_V6_clean-v2__timmu_IMG1312_hrnet_w48-dec256_fl-tr_b4/A_fold-4/test_fnd/solution_building.csv \
    --out_shapefile_dir ../train/output/foundation/A_V6_clean-v2__timmu_IMG1312_hrnet_w48-dec256_fl-tr_b4/A_fold-4/test_fnd/shapefile/ \
    --square_size 5 --simplify_tolerance 0.5 --min_area 5 --percent_positive 0.5

# ensemble 
python baseline/postprocessing/buildings/building_postprocessing.py \
    --foundation_pred_dir ../train/output/foundation/A_V6_clean-v2__timmu_IMG1312_hrnet_w48-dec256_fl-tr_b4/A_fold-4/test_fnd/ \
    --flood_pred_dir ../train/output/flood/3090_V5__timmu_IMG1312_hrnet_w30_fl-tr/3090_fold-4/ensemble/test_fld/ \
    --out_submission_csv ../train/output/foundation/A_V6_clean-v2__timmu_IMG1312_hrnet_w48-dec256_fl-tr_b4/A_fold-4/test_fnd/solution_building.csv \
    --out_shapefile_dir ../train/output/foundation/A_V6_clean-v2__timmu_IMG1312_hrnet_w48-dec256_fl-tr_b4/A_fold-4/test_fnd/shapefile/ \
    --square_size 5 --simplify_tolerance 0.5 --min_area 5 --percent_positive 0.5

# 73.14
python baseline/postprocessing/buildings/building_postprocessing.py \
    --foundation_pred_dir ../train/output/foundation/A_V6_clean-v2__timmu_IMG1312_hrnet_w48-dec256_fl-tr_b4/A_fold-4/test_fnd/ \
    --flood_pred_dir ../train/output/flood/A_V6__timmu_IMG1312_hrnet_w18-dec256_fl-tr_b8/A_fold-4/test_fld/ \
    --out_submission_csv ../train/output/foundation/A_V6_clean-v2__timmu_IMG1312_hrnet_w48-dec256_fl-tr_b4/A_fold-4/test_fnd/solution_building.csv \
    --out_shapefile_dir ../train/output/foundation/A_V6_clean-v2__timmu_IMG1312_hrnet_w48-dec256_fl-tr_b4/A_fold-4/test_fnd/shapefile/ \
    --square_size 5 --simplify_tolerance 0.5 --min_area 5 --percent_positive 0.5

## loss ratio
# 71.33
python baseline/postprocessing/buildings/building_postprocessing.py \
    --foundation_pred_dir ../train/output/foundation/A_V6_clean-v2__timmu_IMG1312_hrnet_w48-dec256_fl-tr_b4/A_fold-4/test_fnd/ \
    --flood_pred_dir ../train/output/flood/A_V6_clean-v3__timmu_IMG1312_hrnet_w18-dec192_fl-tr_b8_ratio0.1/A_fold-4/test_fld/ \
    --out_submission_csv ../train/output/foundation/A_V6_clean-v2__timmu_IMG1312_hrnet_w48-dec256_fl-tr_b4/A_fold-4/test_fnd/solution_building.csv \
    --out_shapefile_dir ../train/output/foundation/A_V6_clean-v2__timmu_IMG1312_hrnet_w48-dec256_fl-tr_b4/A_fold-4/test_fnd/shapefile/ \
    --square_size 5 --simplify_tolerance 0.5 --min_area 5 --percent_positive 0.5

# 69.65
python baseline/postprocessing/buildings/building_postprocessing.py \
    --foundation_pred_dir ../train/output/foundation/A_V6_clean-v2__timmu_IMG1312_hrnet_w48-dec256_fl-tr_b4/A_fold-4/test_fnd/ \
    --flood_pred_dir ../train/output/flood/A_V6_clean-v3__timmu_IMG1312_hrnet_w18-dec192_fl-tr_b8_ratio0.9/A_fold-4/test_fld/ \
    --out_submission_csv ../train/output/foundation/A_V6_clean-v2__timmu_IMG1312_hrnet_w48-dec256_fl-tr_b4/A_fold-4/test_fnd/solution_building.csv \
    --out_shapefile_dir ../train/output/foundation/A_V6_clean-v2__timmu_IMG1312_hrnet_w48-dec256_fl-tr_b4/A_fold-4/test_fnd/shapefile/ \
    --square_size 5 --simplify_tolerance 0.5 --min_area 5 --percent_positive 0.5

# V7 fnd clean 73.9
python baseline/postprocessing/buildings/building_postprocessing.py \
    --foundation_pred_dir ../train/output/foundation/A_V7_FINE_clean-v4__timmu_IMG1312_fl-tr_b4_lrm5e-6/A_fold-4/test_fnd/ \
    --flood_pred_dir ../train/output/flood/3090_V5_clean-v3__timmu_IMG1312_hrnet_w18_fl-tr_b4/3090_fold-4/test_fld/ \
    --out_submission_csv ../train/output/foundation/A_V7_FINE_clean-v4__timmu_IMG1312_fl-tr_b4_lrm5e-6/A_fold-4/test_fnd/solution_building.csv \
    --out_shapefile_dir ../train/output/foundation/A_V7_FINE_clean-v4__timmu_IMG1312_fl-tr_b4_lrm5e-6/A_fold-4/test_fnd/shapefile/ \
    --square_size 5 --simplify_tolerance 0.5 --min_area 5 --percent_positive 0.5

# V7 fnd fld clean 70.0
python baseline/postprocessing/buildings/building_postprocessing.py \
    --foundation_pred_dir ../train/output/foundation/A_V7_FINE_clean-v4__timmu_IMG1312_fl-tr_b4_lrm5e-6/A_fold-4/test_fnd/ \
    --flood_pred_dir ../train/output/flood/A_V7_FINE_clean-v5__timmu_IMG1312_fl-tr_b8_/A_fold-4/test_fld/ \
    --out_submission_csv ../train/output/foundation/A_V7_FINE_clean-v4__timmu_IMG1312_fl-tr_b4_lrm5e-6/A_fold-4/test_fnd/solution_building.csv \
    --out_shapefile_dir ../train/output/foundation/A_V7_FINE_clean-v4__timmu_IMG1312_fl-tr_b4_lrm5e-6/A_fold-4/test_fnd/shapefile/ \
    --square_size 5 --simplify_tolerance 0.5 --min_area 5 --percent_positive 0.5

# V7 fnd 73.89
python baseline/postprocessing/buildings/building_postprocessing.py \
    --foundation_pred_dir ../train/output/foundation/A_V7_FINE_clean-v4__timmu_IMG1312_fl-tr_b4/A_fold-4/test_fnd/ \
    --flood_pred_dir ../train/output/flood/3090_V5_clean-v3__timmu_IMG1312_hrnet_w18_fl-tr_b4/3090_fold-4/test_fld/ \
    --out_submission_csv ../train/output/foundation/A_V7_FINE_clean-v4__timmu_IMG1312_fl-tr_b4/A_fold-4/test_fnd/solution_building.csv \
    --out_shapefile_dir ../train/output/foundation/A_V7_FINE_clean-v4__timmu_IMG1312_fl-tr_b4/A_fold-4/test_fnd/shapefile/ \
    --square_size 5 --simplify_tolerance 0.5 --min_area 5 --percent_positive 0.5