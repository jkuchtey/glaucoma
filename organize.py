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

check = input("MAKE SURE FILE PATHS ARE CORECT !!!!!!!!! TYPE YES TO CLAIM!!!!!")


def move():
    if check == "YES":
        df = pd.read_csv("ORIGA/OrigaList.csv")

        original_path = "ORIGA/Images_Cropped/"
        new_path = "ORIGA_cropped_sorted/"

        zero = df.loc[df["Glaucoma"] == 0]
        one = df.loc[df["Glaucoma"] == 1]

        labels = [zero, one]
        for i in [0, 1]:

            for f in labels[i]["Filename"]:
                print(f)

                oldpath = str(original_path + f)
                print(oldpath)
                if i == 0:
                    newpath = str(new_path + "0/" + f)
                if i == 1:
                    newpath = str(new_path + "1/" + f)
                os.replace(oldpath, newpath)
            

move()
