import pandas as pd

import os


df = pd.read_csv("ORIGA/OrigaList.csv")

zero = df.loc[df["Glaucoma"] == 0]
one = df.loc[df["Glaucoma"] == 1]

for f in one["Filename"]:
    oldpath = str("ORIGA/Images/" + f)
    print(oldpath)
    newpath = str("ORIGA_SORTED/ORIGA_ONE/" + f)
    os.replace(oldpath, newpath)