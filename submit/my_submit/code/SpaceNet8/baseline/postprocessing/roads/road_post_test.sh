#!/bin/bash

EVAL_CSV="../data/Louisiana-West_Test_Public/Louisiana-West_Test_Public_label_image_mapping.csv" # the .csv that prediction was run on
ROAD_PRED_DIR="../train/output/foundation/A_V7_FINE_clean-v4__timmu_IMG1312_fl-tr_b4/A_fold-4/test_fnd" # the directory holding foundation road prediction .tifs. they have suffix _roadspeedpred.tif
FLOOD_PRED_DIR="../train/output/flood/3090_V5_clean-v3__timmu_IMG1312_hrnet_w18_fl-tr_b4/3090_fold-4/test_fld" # the directory holding flood prediction .tifs. They have suffix _floodpred.tif

OUT_SKNW_WKT="${ROAD_PRED_DIR}/sknw_wkt.csv"
GRAPH_NO_SPEED_DIR="${ROAD_PRED_DIR}/graphs_nospeed"
WKT_TO_G_LOG_FILE="${ROAD_PRED_DIR}/wkt_to_G.log"

GRAPH_SPEED_DIR="${ROAD_PRED_DIR}/graphs_speed"
INFER_SPEED_LOG_FILE="${ROAD_PRED_DIR}/graph_speed.log"

SUBMISSION_CSV_FILEPATH="${ROAD_PRED_DIR}/../solution_road.csv" # the name of the submission .csv
OUTPUT_SHAPEFILE_PATH="${ROAD_PRED_DIR}/flood_road_speed.shp"

python baseline/postprocessing/roads/vectorize_roads.py --im_dir $ROAD_PRED_DIR --out_dir $ROAD_PRED_DIR --write_shps --write_graphs --write_csvs --write_skeletons

python baseline/postprocessing/roads/wkt_to_G.py --wkt_submission $OUT_SKNW_WKT --graph_dir $GRAPH_NO_SPEED_DIR --log_file $WKT_TO_G_LOG_FILE --min_subgraph_length_pix 20 --min_spur_length_m 10

python baseline/postprocessing/roads/infer_speed.py --eval_csv $EVAL_CSV --mask_dir $ROAD_PRED_DIR --graph_dir $GRAPH_NO_SPEED_DIR --graph_speed_dir $GRAPH_SPEED_DIR --log_file $INFER_SPEED_LOG_FILE
 
python baseline/postprocessing/roads/create_submission.py --flood_pred $FLOOD_PRED_DIR --graph_speed_dir $GRAPH_SPEED_DIR --output_csv_path $SUBMISSION_CSV_FILEPATH --output_shapefile_path $OUTPUT_SHAPEFILE_PATH

# python baseline/postprocessing/roads/create_submission.py --flood_pred "../train/output/flood/3090_V1_IMG512_effb7_dec512_hard-aug_no-rotate/3090_fold-0/test_fld" --graph_speed_dir "../train/output/foundation/A_V1_IMG512_effb7_dec512/A_fold-0/test_fndgraphs_speed" --output_csv_path "../train/output/foundation/A_V1_IMG512_effb7_dec512/A_fold-0/debug.csv --output_shapefile_path "../train/output/foundation/A_V1_IMG512_effb7_dec512/A_fold-0/test_fnd/flood_road_speed.shp"