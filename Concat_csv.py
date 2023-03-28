import pandas as pd
import glob

dataframe_lst= []
for file in glob.iglob('filepathtothefiles/*'):
    print(file)
    dataframe = pd.read_csv(file, dtype='str')
    dataframe_lst.append(dataframe['text'])
dtframe = pd.concat(dataframe_lst, axis=0, ignore_index=True)

dtframe.to_csv(f'filepath/filename.csv') 
