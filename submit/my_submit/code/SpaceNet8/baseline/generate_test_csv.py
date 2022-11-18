import pandas as pd
from glob import glob
import os
import sys

# test private
args = sys.argv
DATA_ROOT = args[1]


PATH_CSV = glob(f'{DATA_ROOT}/MysteryCity_Test_Private/*_image_mapping.csv')[0]
df = pd.read_csv(PATH_CSV)

DIR_TEST = f'{DATA_ROOT}/MysteryCity_Test_Private/'

pres = []
posts = []

for i, _, pre, post_1, post_2 in df.itertuples():

     PATH_PRE = os.path.join(DIR_TEST, 'PRE-event', pre)
     PATH_POST_1 = os.path.join(DIR_TEST, 'POST-event', post_1)
     

     if os.path.exists(PATH_PRE) and os.path.exists(PATH_POST_1):
          pres.append(PATH_PRE)
          posts.append(PATH_POST_1) 
    
     if isinstance(post_2, str):
          PATH_POST_2 = os.path.join(DIR_TEST, 'POST-event', post_2)
          if os.path.exists(PATH_PRE) and os.path.exists(PATH_POST_2):
               pres.append(PATH_PRE)
               posts.append(PATH_POST_2)
               
test_df = pd.DataFrame()
test_df['preimg'] = pres
test_df['postimg'] = posts
test_df.head()

PATH_TEST_SAVE_CSV = f'./test_preimg-postimg.csv'
test_df.to_csv(PATH_TEST_SAVE_CSV, index=False, header=True)