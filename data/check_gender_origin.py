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

def write_discard(item):
    with open('discard_new.txt', 'a') as f:
            f.write("%s\n" % item)
def write_old(item):
    with open('discard_age.txt', 'a') as f:
            f.write("%s\n" % item)
client = Client()

q_numbers=[x.replace(".txt", "") for x in os.listdir("text/")]
print("Already having {} entities in the folder".format(len(q_numbers)))
done=[x.replace(".txt", "") for x in os.listdir("man/text/")]
done+=[x.replace(".txt", "") for x in os.listdir("woman/text/")]
done+=[x.replace(".txt", "") for x in os.listdir("other/text")]
with open("discard_age.txt") as f:
  too_old = f.readlines()
too_old = [element.rstrip('\n') for element in too_old]
discard = []

with open("discard_new.txt") as f:
  discard = f.readlines()
discard = [element.rstrip('\n') for element in discard]

q_numbers = [number for number in q_numbers if number not in too_old] 
q_numbers = [number for number in q_numbers if number not in discard] 
q_numbers = [number for number in q_numbers if number not in done] 

print("Using {} entities ".format(len(q_numbers)))

for q_number in tqdm.tqdm(q_numbers):
    try:
        entity = client.get(q_number, load=True)
        description = str(entity.description)
        name = str(entity.label)
        age = int(entity.attributes["claims"]["P569"][0]["mainsnak"]["datavalue"]["value"]["time"][1:5]) 
        if age < 1900:
            too_old.append(q_number)
            write_old(q_number)
            continue
        age = str(age)
        gender = entity.attributes["claims"]["P21"][0]["mainsnak"]["datavalue"]["value"]["id"]
        country = entity.attributes["claims"]["P27"][0]["mainsnak"]["datavalue"]["value"]["id"]
        country_client = client.get(country, load=True)
        continent = country_client.attributes["claims"]["P30"][0]["mainsnak"]["datavalue"]["value"]["id"]
    except KeyError as e:
        print("[Error] ", e, q_number)
        discard.append(q_number)
        write_discard(q_number)
        continue
    
    if gender == "Q6581097": # male
        subprocess.call(f"ln -s ../../images/{q_number}.jpg man/images/{q_number}.jpg", shell=True)
        with open("man/text/{}.txt".format(q_number), "w") as f:
            f.write("he " + description + " "  + continent + " " +  age)
    elif gender == "Q6581072": #female
        subprocess.call(f"ln -s ../../images/{q_number}.jpg woman/images/{q_number}.jpg", shell=True)
        with open("woman/text/{}.txt".format(q_number), "w") as f:
            f.write("she " + description + " "  + continent + " " + age)
    else: # other gender
        subprocess.call(f"ln -s ../../images/{q_number}.jpg other/images/{q_number}.jpg", shell=True)
        with open("other/text/{}.txt".format(q_number), "w") as f:
            f.write("it " + description + " "  + continent+ " " + age)
