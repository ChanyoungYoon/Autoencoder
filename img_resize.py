import os
import glob
from PIL import Image

files = './dataset/metal_nut_test/'
copy_path = './dataset/metal_nut_test/'
for f in os.listdir(files):
    img = Image.open(files + f)
    img_resize = img.resize((512, 512))
    title, ext = os.path.splitext(f)
    img_resize.save(copy_path + title + ext)