import pandas as pd 
import os 
import glob 
from tqdm import tqdm 

data_path = glob.glob('./openwebtext/*.parquet')[0]
save_path = './openwebtext/text'

if not os.path.exists(save_path):
    os.makedirs(save_path)

table = pd.read_parquet(data_path)

num_rows = table.shape[0]
print(num_rows)

nums_per_file = 10000

for i in tqdm(range(0,num_rows,nums_per_file)):
    file_name = f'{save_path}/file_{i//nums_per_file}.txt'

    with open(file_name,'w') as f:
        for row in range(i, min(i+nums_per_file, num_rows)):
            f.write(str(table.loc[row]['text']) + '\n')