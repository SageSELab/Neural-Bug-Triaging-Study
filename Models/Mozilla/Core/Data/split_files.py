
import pandas as pd

def split_csv_file(input_file_path):
    for i,chunk in enumerate(pd.read_csv(f'./{input_file_path}.csv', chunksize=15000)):
        chunk.to_csv(f'{input_file_path}_{i}.csv', index=False)

split_csv_file('mozilla_core_fold3_train')
split_csv_file('mozilla_core_fold4_train')
split_csv_file('mozilla_core_fold5_train')
split_csv_file('mozilla_core_fold6_train')
split_csv_file('mozilla_core_fold7_train')
split_csv_file('mozilla_core_fold8_train')
split_csv_file('mozilla_core_fold9_train')
split_csv_file('mozilla_core_fold10_train')