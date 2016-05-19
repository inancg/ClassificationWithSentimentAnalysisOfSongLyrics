import requests
import json
import glob

smiths_songs = (glob.glob("./Moz/*"))
url = "http://text-processing.com/api/sentiment/"

def get_label(line):
  req = requests.post(url, "text="+line,headers={'Content-Type': 'application/json'})
  result = req.json()
  return result['label']

def get_label_for_song(file_name):
  pos = 0
  neg = 0
  neut = 0
  f = open(file_name,'r')
  print (file_name)
  for line in f:
     req = line.rstrip()
     if (len(req) > 0):
       #print (req)
       label = get_label(req)
       #print (label)
       if (label == 'pos'): pos += 1
       elif (label == 'neg'): neg += 1
       elif (label == 'neutral'): neut += 1
       print (pos,neg,neut)
  result = 0
  if max([pos,neg,neut])==pos:
  	result = 1
  elif max([pos,neg,neut])==neg:
  	result = -1
  else : result = 0
  return [result,pos,neg,neut]

result = []
for song in smiths_songs:
	tmp = get_label_for_song(song)
	result.append(tmp)

print(result)