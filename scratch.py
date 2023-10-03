import os
import shutil

for file in os.listdir("./models"):

    if os.path.isdir(f"models/{file}"):
        continue
    elif os.path.basename(file).endswith(".pth"):
        name = os.path.basename(file)
        folder = f"models/{name.split('.')[0]}"
        shutil.move(f"models/{name}", folder)