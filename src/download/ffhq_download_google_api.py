import json
import requests
import threading
import logging
import os

from tqdm import tqdm
import os.path
import io

from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseDownload


json_file = "/home/lucas/Documentos/Image-Denoising/ffhq-dataset-v2.json"
scopes = ["https://www.googleapis.com/auth/drive"]
token = "/home/lucas/Documentos/Image-Denoising/token.json"

creds = Credentials.from_authorized_user_file(token, scopes)
service = build("drive", "v3", credentials=creds)

with open(json_file,"r") as infile:
    load = json.load(infile)
    infile.close()


for key_ in tqdm(load.keys()):
    path = "/".join(load[key_]['image']['file_path'].split("/")[:-1])
    os.makedirs(path,exist_ok=True)

def download(start,end):
    for idx in range(start,end):
        idx = str(idx)
        path = load[idx]['image']['file_path']
        if(not os.path.exists(path)):
            url = load[idx]['image']['file_url']
            position = url.find('id=')
            url = url[position:].replace('id=','')
            
            request = service.files().get_media(fileId=url)
            
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()

            with open(path,'wb') as outfile:
                fh.seek(0)
                outfile.write(fh.read())
                outfile.close()

            print(f"Image {path} downloaded")
        else:
            print(f"Image {path} already exists")


for idx in tqdm(load.keys()):
    path = load[idx]['image']['file_path']
    if(not os.path.exists(path)):
        url = load[idx]['image']['file_url']
        position = url.find('id=')
        url = url[position:].replace('id=','')
            
        request = service.files().get_media(fileId=url)
            
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
            
        done = False
        while done is False:
            status, done = downloader.next_chunk()

        with open(path,'wb') as outfile:
            fh.seek(0)
            outfile.write(fh.read())
            outfile.close()

if False:
    total = len(load)
    partes = 1
    
    
    tamanho_parte = total // partes
    
    threads = []
    
    logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%H:%M:%S")
    
    for index, i in enumerate(range(partes)):
        start, end = (i * tamanho_parte, (i + 1) * tamanho_parte - 1)
        logging.info("Main    : create and start thread %d.", index)
        x = threading.Thread(target=download, args=(start,end,))
        threads.append(x)
        x.start()
