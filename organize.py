import pandas as pd

import os


df = pd.read_csv("ORIGA/OrigaList.csv")

zero = df.loc[df["Glaucoma"] == 0]
one = df.loc[df["Glaucoma"] == 1]

for f in zero["Filename"]:
    print(f)
    oldpath = str("ORIGA/Images_Cropped/" + f)
    print(oldpath)
    newpath = str("ORIGA_cropped_sorted/0/" + f)
    os.replace(oldpath, newpath)