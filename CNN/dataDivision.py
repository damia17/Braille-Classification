# Devide the dataset putting the images of each letter in a folder with the name of the letter

import os
import numpy as np
import pandas as pd
from shutil import copyfile


os.mkdir('./dataImages/')
alphabet = 'a'
for i in range(26):
    os.mkdir('./dataImages/' + alphabet)
    alphabet = chr(ord(alphabet) + 1)
root = "./Braille Dataset/Braille Dataset/"
for f in os.listdir(root):
    letter = f[0]
    copyfile(root + f, './dataImages/' + letter + '/' + f)