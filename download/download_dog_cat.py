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
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(t, path=dow_dir)
    os.remove(filename)
