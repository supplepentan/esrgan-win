import os
import requests
import zipfile
import tarfile

urls = [
    "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
    "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
]

dow_dir = "input"
os.makedirs(dow_dir, exist_ok=True)
for url in urls:
    urlData = requests.get(url).content
    filename = os.path.basename(url)
    with open(filename, mode='wb') as f:
        f.write(urlData)
    with tarfile.open(filename) as t:
        t.extractall(path=dow_dir)
    os.remove(filename)
