import sys
import os

# test private
args = sys.argv
SUB_ROOT = args[1]


ROAD_CSV = '../../train/output/foundation/A_V7_FINE_clean-v4__timmu_IMG1312_fl-tr_b4/A_fold-4/solution_road.csv'
BUILD_CSV = '../../train/output/foundation/A_V7_FINE_clean-v4__timmu_IMG1312_fl-tr_b4/A_fold-4/test_fnd/solution_building.csv'
SUBMIT_SCV = os.path.join('/workdir/', SUB_ROOT, 'solution.csv')


import pandas as pd

# submit_colnmns = ['ImageId', 'Object', 'Flooded', 'Wkt_Pix', 'Wkt_Geo']
submit_colnmns = ['ImageId', 'Object', 'WKT_Pix', 'Flooded']

dfs = []

df = pd.read_csv(ROAD_CSV)
df = df[submit_colnmns + ['length_m', 'travel_time_s']]
dfs.append(df)

df = pd.read_csv(BUILD_CSV).rename(columns={'Wkt_Pix': 'WKT_Pix', 'Wkt_Geo': 'WKT_Geo'})
df = df[submit_colnmns]
df['length_m'] = 'null'
df['travel_time_s'] = 'null'
dfs.append(df)
    
df = pd.concat(dfs)

# 重複削除
check_col = submit_colnmns + ['length_m', 'travel_time_s']

df_dup = df.duplicated(subset=check_col)
num_dup = len(df_dup == False)
print(f"no duplicated: {num_dup}")
df_sub = df[~df_dup]

df_sub.head(3)

def remove_duplicated(linestring):
    
    if linestring != 'POLYGON EMPTY':
        # exist object
        if 'LINESTRING' in linestring:
            points = linestring.split('(')[1].split(')')[0]
            # print(points)
            
            knowns = []
            for i, xy in enumerate(points.split(',')):
                if i == 0:
                    knowns.append(' ' + xy)
                else:
                    if not xy in knowns:
                        knowns.append(xy) 
                    else:
                        print('◆'*20, f'delete duplicated {xy} <<< {linestring}')
            
            # top space
            knowns[0] = knowns[0][1:]
            linestring = 'LINESTRING (' + ','.join(knowns) + ')'
    return linestring
               
df_sub['WKT_Pix'] = df_sub['WKT_Pix'].apply(remove_duplicated)
df_sub.head(3)

df_sub.to_csv(SUBMIT_SCV, index=False, header=True)