#!/usr/bin/env python
# coding: utf-8
from wikidata.client import Client
import matplotlib.pyplot as plt
import numpy as np
import json, os
import tqdm
import urllib.request
import time
import pdb, subprocess

#q_numbers=[x.replace(".txt", "") for x in os.listdir("man/text/")]
q_numbers=[x.replace(".txt", "") for x in os.listdir("woman/text/")]
#q_numbers+=[x.replace(".txt", "") for x in os.listdir("other/text/")]

print("Having {} entities in the folder".format(len(q_numbers)))


def q_to_continent(Q):
    if Q == "Q15":
        return "africa"
    elif Q == "Q49":
        return "canada"
    elif Q == "Q5401":
        return "eurasia"
    elif Q == "Q46":
        return "europe"
    elif Q == "Q538":
        return "australia"
    elif Q == "Q18":
        return "brazil"
    elif Q == "Q48":
        return "asia"
    else:
        print(f"Something is wrong, {Q}")

for q_number in tqdm.tqdm(q_numbers):
        subprocess.call(f"ln -s ../../images/{q_number}.jpg all/images/{q_number}.jpg", shell=True)
        with open(f"woman/text/{q_number}.txt", "r") as f:
            line = f.readlines()
            year = int(line[0].split()[-1])
            # categorize
            if year <= 1942:
                age = "old"
            elif (year >= 1943) and (year <= 1957):
                age = "middle"
            else: 
                age = "young"
            tmp = line[0].split()
            tmp[-1] = age
            continent = q_to_continent(tmp[-2])
            tmp[-2] = continent
            with open(f"all/text/{q_number}.txt", "w") as foo:
                foo.write(" ".join(tmp))
