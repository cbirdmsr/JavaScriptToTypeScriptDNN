import os
import csv
import requests
import glob
from shutil import copyfile
from subprocess import call

in_file = "repos.csv"
clone_dir = "Clones"
aligned_file = "aligned.txt"

if not os.path.exists(in_file):
  page = 0
  base_query = "https://api.github.com/search/repositories"
  repos = []
  for i in range(10):
    params = {"q": "  language:typescript fork:false", "sort": "stars", "order": "desc", \
            "pushed": ">=2016-01-01", "per_page": 100, \
            "page": i }
    response = requests.get(base_query, params)
    print(response.url)
    json = response.json()
    repos += json.get("items")

  with open(in_file, "w") as f:
    f.write("name,language,stars,forks,size,last_pushed\n")
    for r in repos:
      f.write(r.get("full_name"))
      f.write(",")
      f.write(r.get("language"))
      f.write(",")
      f.write(str(r.get("stargazers_count")))
      f.write(",")
      f.write(str(r.get("forks")))
      f.write(",")
      f.write(str(r.get("size")))
      f.write(",")
      f.write(str(r.get("pushed_at")))
      f.write("\n")

with open(in_file, "r") as f:
  reader = csv.reader(f, delimiter=",")
  lines = [line for line in reader]
  lines = lines[1:]
  
if not os.path.exists(clone_dir):
  os.mkdir(clone_dir)
  for line in lines:
    name = line[0]
    print(name)
    parent = name[:name.index("/")]
    child = name[name.index("/") + 1:]
    parent_dir = clone_dir + "/" + parent
    child_dir = parent_dir + "/" + child
    if not os.path.exists(parent_dir):
      os.mkdir(parent_dir)
    github_url = "https://github.com/" + name + ".git"
    command = "git clone " + github_url + " " + child_dir
    os.system(command)
