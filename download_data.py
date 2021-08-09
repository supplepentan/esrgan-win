import os
import requests
import zipfile
import tarfile

"""
if(not os.path.exists("./input")): os.makedirs("input")
url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
filepath = "input/images.tar.gz"
urlData = requests.get(url).content
with open(filepath, mode='wb') as f:
    f.write(urlData)
with tarfile.open(filepath) as t:
    t.extractall(path="input")
os.remove(filepath)


url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
print(os.path.basename(url))

"""

urls = [
    #"https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
    "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
    ]

for url in urls:
    urlData = requests.get(url).content
    filename = os.path.basename(url)
    with open(filename, mode='wb') as f:
        f.write(urlData)
    with tarfile.open(filename) as t:
        t.extractall(path="input")
    os.remove(filename)