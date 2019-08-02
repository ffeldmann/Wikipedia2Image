#!/usr/bin/env python
# coding: utf-8
import numpy as np
import json, os
import tqdm
import urllib.request
import time
import pdb
import subprocess
import skimage.io
import warnings
warnings.filterwarnings("error")


q_numbers=[x.replace(".txt", "") for x in os.listdir("text_shape/")]
print("Having {} entities in the folder".format(len(q_numbers)))

with open("../query_woman.json", "r") as f:
   query_woman = json.load(f)

with open("../query_man.json", "r") as f:
   query_man = json.load(f)

with open("../query_100_all.json", "r") as f:
   query_100 = json.load(f)

def check_100(q_number):
    for element in query_100:
        if element['item'].replace("http://www.wikidata.org/entity/","") == q_number:
            return True
        else:
            return False

def check_gender(q_number):
    for element in query_woman:
        element =  element['item'].replace("http://www.wikidata.org/entity/","")
        if element == q_number:
            # check if in 100
            if check_100(element):
                return "woman"
            return False
    # does not seem to be a woman, but is it a man?
    for element in query_man:
        element =  element['item'].replace("http://www.wikidata.org/entity/","")
        if element == q_number:
            # check if in 100
            if check_100(element):
                return "man"
            return False
man = 0
woman = 0

for element in tqdm.tqdm(q_numbers):
    answer = check_gender(element)
    if answer == "woman":
        woman +=1
        print("WOMAN")
        subprocess.call(f"ln -s images/{element}.jpg images_woman/{element}.jpg", shell=True)
        subprocess.call(f"ln -s text/{element}.txt text_woman/{element}.txt", shell=True)
    elif answer == "man":
        man +=1
        print("MAN")
        subprocess.call(f"ln -s text/{element}.txt text_man/{element}.txt", shell=True)
        subprocess.call(f"ln -s images/{element}.jpg images_man/{element}.jpg", shell=True)
    else:
        pass
        #print("Not in 100 or neither man or woman")

print(f"Man {man}, Woman {woman}")
