import torch
import numpy as np
import pandas as pd
import os


data_dir = "data_factory/data"

file_address = os.path.join(data_dir, "delaney.csv")
df = pd.read_csv(file_address)
df = df.sample(frac=1, random_state=0).reset_index(drop=True)
df_train = df.iloc[:700]
df_valid = df.iloc[700:900]
df_test = df.iloc[900:]

df_train.to_csv(os.path.join(data_dir, "delaney_train.csv"), index=False)
df_valid.to_csv(os.path.join(data_dir, "delaney_valid.csv"), index=False)
df_test.to_csv(os.path.join(data_dir, "delaney_test.csv"), index=False)
