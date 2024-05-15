import pandas as pd

import os


df = pd.read_csv("ORIGA/OrigaList.csv")

zero = df.loc[df["Glaucoma"] == 0]
one = df.loc[df["Glaucoma"] == 1]

for f in one["Filename"]:
    print(f)
    oldpath = str("ORIGA/Images_Square/" + f)
    print(oldpath)
    newpath = str("ORIGA_square_sorted/1/" + f)
    os.replace(oldpath, newpath)