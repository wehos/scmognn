# Copyright 2022 DSE lab.  All rights reserved.
import requests 
import os
import zipfile 
import tqdm
import urllib.request


# judge whether a folder exits or not, if not, to create a folder
def judge_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

# delete zip file
def delete_file(filename):
    if not os.path.exists(filename):
        print("File does not exist")
    else:
        print("Deleting", filename)
        os.remove(filename)

# download zip file from url with progress bar
def download_file(url, filename):
    judge_folder(os.path.dirname(filename))
    if not os.path.exists(filename):
        print("Downloading", filename)
        u = urllib.request.urlopen(url)
        f = open(filename, 'wb')
        meta = u.info()
        file_size = int(meta.get("Content-Length"))
        print("Downloading: %s Bytes: %s" % (filename, file_size))

        file_size_dl = 0
        block_sz = 8192
        with tqdm.tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024) as bar:
            while True:
                buffer = u.read(block_sz)
                if not buffer:
                    break
                file_size_dl += len(buffer)
                f.write(buffer)
                bar.update(len(buffer))
        f.close()
        u.close()
        return True
    else:
        print("File already downloaded")
        return False

# unzip zipfile
def unzip_file(filename, directory_to_extract_to):
    if not os.path.exists(filename):
        print("File does not exist")
    else:
        print("Unzipping", filename)
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)
        delete_file(filename)
