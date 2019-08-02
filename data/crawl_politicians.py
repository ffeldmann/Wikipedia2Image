#!/usr/bin/env python
# coding: utf-8
from wikidata.client import Client
import matplotlib.pyplot as plt
import numpy as np
import json, os
import tqdm
import urllib.request
import time
import pdb
from wikipedia_crawler import crawler
WAIT = 100
WAIT_TIME = 2


with open("query.json", "r+") as f:
    data = json.load(f)

client = Client()

#entities = {} # entitynumner: (imgurl, texturl)
try:
    with open("discard.txt", "r") as f:
        discard = [line.rstrip("\n") for line in f]
except:
    discard = []
done = []
def write_discard(item):
    with open('discard.txt', 'a') as f:
            f.write("%s\n" % item)

done=[x.replace(".txt", "") for x in os.listdir("wikidata_politicans/"+"text/")]
print("Already having {} entities in the folder".format(len(done)))
print("Removing already downloaded entities....")
data[:] = [d for d in data if d.get('item').replace("http://www.wikidata.org/entity/","") not in done]
data[:] = [d for d in data if d.get('item').replace("http://www.wikidata.org/entity/","") not in discard]

print("New length: {}".format(len(data)))
num = 0
for element in tqdm.tqdm(data):
    entityID = element['item'].replace("http://www.wikidata.org/entity/","")
    # if entity is present already then continue
    if entityID in done or entityID in discard:
        continue
    try:
        entity = client.get(entityID, load=True)
        text_url = entity.data['sitelinks']['enwiki']['url']
        image_prop = client.get('P18')
        image = entity[image_prop]
        image.load()
    except KeyError as e:
        #print("[Error] ", e)
        discard.append(entityID)
        write_discard(entityID)
        continue
    
    try:
        image_url = image.data['imageinfo'][0]['url']
    except:
        discard.append(entityID)
        write_discard(entityID)
        continue
    #entities[entityID] = (image_url, text_url)
    
    # get the data
    crawler(text_url, output_file="wikidata_politicans/text/{}.txt".format(entityID))
    urllib.request.urlretrieve(image_url, "wikidata_politicans/images/{}.jpg".format(entityID))
    done.append(entityID)
    num +=1
    if num % WAIT == 0:
        time.sleep(WAIT_TIME)

