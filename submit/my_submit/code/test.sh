# !/bin/bash
echo "test start args is:" "$1" "$2" 

cd /workdir/SpaceNet8/baseline/

# prepare `mapping.csv -> inference list`
python generate_test_csv.py $1

# inference
python inference_foundation.py
python inference_flood.py

cd /workdir/SpaceNet8/
# postprocess
./baseline/postprocessing/roads/road_post_test.sh
python baseline/postprocessing/buildings/building_postprocessing.py \
    --foundation_pred_dir ../train/output/foundation/A_V7_FINE_clean-v4__timmu_IMG1312_fl-tr_b4/A_fold-4/test_fnd/ \
    --flood_pred_dir ../train/output/flood/3090_V5_clean-v3__timmu_IMG1312_hrnet_w18_fl-tr_b4/3090_fold-4/test_fld/ \
    --out_submission_csv ../train/output/foundation/A_V7_FINE_clean-v4__timmu_IMG1312_fl-tr_b4/A_fold-4/test_fnd/solution_building.csv \
    --out_shapefile_dir ../train/output/foundation/A_V7_FINE_clean-v4__timmu_IMG1312_fl-tr_b4/A_fold-4/test_fnd/shapefile/ \
    --square_size 5 --simplify_tolerance 0.5 --min_area 5 --percent_positive 0.5

cd /workdir/SpaceNet8/baseline/
# submit
python make_submit.py $2

echo 'done'


 
