import os
import cv2
from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import PIL.Image as Image
import torch

def get_sinusoid_encoding(num_tokens, token_len):
    """Make Sinusoid Encoding Table
    
    Args:
        num_tokens (int): number of tokens
        token_len (int): length of a token
                
    Returns:
        torch.FloatTensor: sinusoidal position encoding table
    """
    def get_position_angle_vec(i):
        """Calculate the positional angle vector for a given position i"""
        return [i / np.power(10000, 2 * (j // 2) / token_len) for j in range(token_len)]
    
    # Create a sinusoid table with positional angle vectors for each token
    sinusoid_table = np.array([get_position_angle_vec(i) for i in range(num_tokens)])
    
    # Apply sine to even indices in the array; 2i
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    
    # Apply cosine to odd indices in the array; 2i+1
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    
    # Convert the numpy array to a torch FloatTensor and add a batch dimension
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class FindFacesGeneric():
    def __init__(self,path_dir):
        self.path_dir = path_dir
        self.pointers = {}
        self.start()

    def start(self):
        list_pointers = []
        for root_, dir_, files_ in os.walk(self.path_dir):
            if(len(files_) > 0):
                for file_ in files_:
                    if(file_.endswith('pts') and "checkpoint.pts" != file_[-14:]):
                        list_pointers.append(f"{root_}/{file_}")

        for pointer in list_pointers:
            number = int(pointer.split('_')[-1].replace('.pts',''))

            key = str(pointer[:-7]) if number > 9 else str(pointer[:-6])

            self.pointers[key.split("/")[-1]] = {}

        for pointer_name in tqdm(list_pointers):
            pointer_name_split = pointer_name.split('_')
            key = pointer_name_split[0]
            number = pointer_name_split[1].replace('.pts','')

            self.pointers[key.split("/")[-1]][number] = pointer_name
    
    def extract_landmark(self,path:str)->list:
        if(not os.path.exists(path)):
            raise Exception(f"Path does not exists: {path}")

        landmark_pointer = []

        with open(path, 'r') as infile:
            # Skip the first three lines
            for _ in range(3):
                infile.readline()

            # Read the landmark points until the closing brace is found
            line = infile.readline().strip()
            while line != '}':
                landmark_pointer.append([int(float(x)) for x in line.split()])
                line = infile.readline().strip()
        
        return landmark_pointer

    def rescale_landmark_crop(self,landmarks:list,image,rescale_face_cropped:float):
        # Convert the landmark points to a NumPy array and calculate the bounding box
        aux_tranpose = np.array(landmarks).T
        x_min, y_min = aux_tranpose[0].min(), aux_tranpose[1].min()
        x_max, y_max = aux_tranpose[0].max(), aux_tranpose[1].max()

        # Calculate the amount to expand the bounding box by the rescale factor
        var_x = int((x_max - x_min) * (rescale_face_cropped - 1) / 2)
        var_y = int((y_max - y_min) * (rescale_face_cropped - 1) / 2)

        # Crop the face from the image
        img = np.array(image.copy())
        face_ = img[y_min - var_y:y_max + var_y, x_min - var_x:x_max + var_x].copy()

        return face_


    def save_face_images(self,path_to_save:str,rescale_face=1,size=(224,224)):

        for key in tqdm(self.pointers.keys()):
            keys = list(self.pointers[key].keys())
            try:
                image_path = self.pointers[key][keys[0]].replace(".pts",".jpg")
            except:
                print(self.pointers[key])
                raise Exception("")
            if(not os.path.exists(image_path)):
                raise Exception(f"Path does not find: {image_path}")
            
            img = Image.open(image_path).convert("RGB")
            
            for key_ in keys:
                name = f"{path_to_save}/{key}_{key_}.png"
                land = self.extract_landmark(self.pointers[key][key_])
                res = self.rescale_landmark_crop(land,img,rescale_face)
                res = cv2.cvtColor(res,cv2.COLOR_BGR2RGB)
                res = cv2.resize(res, size)
                cv2.imwrite(name,res)


class FindFaces300W(FindFacesGeneric):
    def __init__(self,path_dir):
        self.path_dir = path_dir
        self.pointers = []
        self.start()

    def start(self):
        list_pointers = []
        for root_, dir_, files_ in os.walk(self.path_dir):
            if(len(files_) > 0):
                for file_ in files_:
                    if(file_.endswith('pts') and "checkpoint.pts" != file_[-14:]):
                        list_pointers.append(f"{root_}/{file_}")
        
        self.pointers = list_pointers

    def save_face_images(self,path_to_save:str,rescale_face=1,size=(224,224)):

        for pointer_path in tqdm(self.pointers):
            image_path = pointer_path.replace(".pts",".png")
            
            if(not os.path.exists(image_path)):
                raise Exception(f"Path does not find: {image_path}")
            
            img = Image.open(image_path).convert("RGB")
            
            name = f"{path_to_save}/{'_'.join(image_path.split('/')[-2:])}"
            land = self.extract_landmark(pointer_path)
            res = self.rescale_landmark_crop(land,img,rescale_face)
            res = cv2.cvtColor(res,cv2.COLOR_BGR2RGB)
            res = cv2.resize(res, size)
            cv2.imwrite(name,res)



class PlotGrapich():
    def __init__(self):
        self.histogram = None

    def histogram_from_poiters(self,pointers):
        aux = []
        for key in pointers.keys():
            aux.append(len(pointers[key]))

        df_histogram = pd.Series(aux).value_counts().to_frame().reset_index()
        df_histogram.rename({"index":"qtd_face"},axis=1,inplace=True)
        self.histogram = df_histogram

    def plot_histogram(self,df_histogram=None,figsize=(10,5)):

        if(df_histogram is None):
            df_histogram = self.histogram

        fig, ax = plt.subplots(figsize=figsize)
        bar = sns.barplot(data=df_histogram,x="qtd_face",y="count",ax=ax)

        for container in bar.containers:
            ax.bar_label(container, fmt='%d')

        ax.set_xlabel("Number of faces")
        ax.set_ylabel("")
        ax.set_yticks([])

        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        plt.show()

    def compare_histograms(self,histograms:list,figsize=(10,5)):
        for df, label in histograms:
            df["Dataset"] = label

        df = pd.concat([df_ for df_,_ in histograms])
        
        df["count"] = df["count"].apply(lambda x: np.log(x) + 1)
        
        fig, ax = plt.subplots(figsize=figsize)
        bar = sns.barplot(data=df,x="qtd_face",y="count",hue="Dataset",ax=ax)
        
        for container in bar.containers:
            ax.bar_label(container, fmt=lambda x: "{}".format(int(round(np.exp(x-1)))))
        
        ax.set_xlabel("Number of faces")
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.set_yticks([])
        
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        
        plt.show()


