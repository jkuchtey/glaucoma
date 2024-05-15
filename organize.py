import pandas as pd

import os

g1020 = pd.read_csv("G1020/G1020.csv")
g1020zero = g1020.loc[g1020["Glaucoma"] == 0]
df = pd.read_csv("ORIGA/OrigaList.csv")

# zero = df.loc[df["Glaucoma"] == 0]
# one = df.loc[df["Glaucoma"] == 1]

# for f in one["Filename"]:
#     oldpath = str("ORIGA/Images_Square/" + f)
#     newpath = str("ORIGA_square_sorted/1/" + f)
#     os.replace(oldpath, newpath)

# for f in zero["Filename"]:
#     oldpath = str("ORIGA/Images_Square/" + f)
#     newpath = str("ORIGA_square_sorted/0/" + f)
#     os.replace(oldpath, newpath)

