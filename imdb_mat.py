import numpy as np
import cv2
from tqdm import tqdm
from utils import get_meta
#import functions as ft
import scipy.io
#Serializer Imdb-wiki

# Infos Importantes:
'''
faceScore -> confiança do detector de faces nas imagens
FullPath-> Diretorio da imagem na base
gender -> genero das pessoas
Age -> idade (para classificação-> Intervalos de idade| regressão-> formatação atual)
dob-> data de nasci(n é necessário para o pre-process)
secondfacescore-> imagens em q há mais de um rosto detectado(descartar)


'''

db="imdb"
root_path = "{}_crop/".format(db)
mat_path = root_path + "{}.mat".format(db) 
min_score = 1.00 #Minimo de Confiança do detector de faces
img_size = 64 #Resolução minima


out_genders = []
out_ages = []
out_imgs = []
full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(mat_path, db)


for i in tqdm(range(len(face_score))):
      if face_score[i] < min_score:
            continue

      if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
            continue
        
      if ~(0 <= age[i] <= 100):
            continue

      if np.isnan(gender[i]):
            continue


      out_genders.append(int(gender[i]))
      #catAge = ft.imdbAge(age[i])
      out_ages.append(age[i])
      img = cv2.imread(root_path + str(full_path[i][0]))
      im =  out_imgs.append(cv2.resize(img, (img_size, img_size)))
     
        
output = {"image": np.array(out_imgs), "gender": np.array(out_genders), "age": np.array(out_ages)}
scipy.io.savemat("serializer/MATimdb64.mat", output)      