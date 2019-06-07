#gender functions pre-process
import numpy as np  # manipular vetores
import pickle as pkl # comprimir/serializar vetores
from PIL import Image # maior eficiência ao processar as imagens
import matplotlib.pyplot as plt  # plotar imagens
from scipy.io import loadmat #Carregar Base de arquivo .mat
import os
import cv2


WIDTH = HEIGHT = 256
MASCULINO = 0
FEMININO = 1


def save(obj,name):
    ''' Função para salvar objeto '''
    with open(name+'.pkl', 'wb') as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)

def load(name):
    ''' Função para carregar objeto serializado '''
    with open(name+'.pkl', 'rb') as f:
        return pkl.load(f)

def show(num_dados,data):
    print('Exemplos de %d imagens da base de treino' % num_dados)
    
    # Escolhemos índices aleatórios
    random_indices = np.random.randint(0, data['imagens'].shape[0], num_dados)
    
    # Buscando imagens e labels
    imagens = data['imagens'][random_indices]
    labels = data['labels'][random_indices]
    
    # Plottando imagens
    for index, (img, label) in enumerate(zip(imagens, labels)):
        plt.subplot(2, num_dados, index + 1)
        plt.axis('off')
        plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('%i' % label)
    plt.show()
    
def test():
   print ("Function OK!")
   

def serializer(fold,df,TypeGA):
    ''' Salvar em arquivo com imagens (formato array numpy) e suas respectivas labels '''
    i = 0
    j = 0
    for f in fold:
        imagens = []
        labels = []
        for index, row in df[j].iterrows():
            print(row)
            user_id = row['user_id']
            original_image = row['original_image']
            face_id = row['face_id']
            
            if(TypeGA==1):
                genero = MASCULINO if row['gender'] == 'm' else FEMININO
                lbl = np.array(genero)
            else:
                idade = categorical_fmt(row['age'])
                lbl = np.array(idade)
                
            # leitura da imagem
            image = Image.open('aligned/aligned/'+user_id+'/landmark_aligned_face.'+str(face_id)+'.'+original_image)
            # redimensiona imagem
            image = image.resize((WIDTH, HEIGHT), Image.LANCZOS)
            image = image.crop((14, 14, 241,241))
            # transforma em array numpy
            img = np.array(image, dtype=np.float16)
            # subtracao da media
            #img /=255 
            img -= np.mean(img)
            # append data
            imagens.append(img)
            labels.append(lbl)
        # save dictionary
        obj = { 'imagens': np.array(imagens), 'labels': np.array(labels) }
        
        if(TypeGA == 1):
            save(obj, 'serializer/gender2/data_'+str(i))
        else:  
            save(obj, 'serializer/age/data_'+str(i))
            
        i += 1
        j += 1
    j=0
    print("Objetos Serializados!")




def fmt1(idade):
    ''' Formata um valor inteiro em um intervalo de idade válido '''
    idade = int(idade)
    if idade >= 0 and idade <= 2:
        return '(0, 2)'
    elif idade >= 4 and idade <= 6:
        return '(4, 6)'
    elif idade >= 8 and idade <= 13:
        return '(8, 13)'
    elif idade >= 15 and idade <= 20:
        return '(15, 20)'
    elif idade >= 25 and idade <= 32:
        return '(25, 32)'
    elif idade >= 38 and idade <= 43:
        return '(38, 43)'
    elif idade >= 48 and idade <= 53:
        return '(48, 53)'
    elif idade >= 60 and idade <= 100:
        return '(60, 100)'
    else:
        return None
    
    
def imdbAge(idade):
    idade = int(idade)
    if 0 <= idade <= 4:
        return 0
    elif 5 <= idade <= 10:
        return 1
    elif  11<= idade <= 15:
        return 2
    elif  16<= idade <= 21:
        return 3
    elif  22<= idade <= 28:
        return 4
    elif  29<= idade <= 39:
        return 5
    elif  40<= idade <= 50:
        return 6
    elif  51<= idade <= 60:
        return 7
    elif  61<= idade <= 100:
        return 8
    
        
    

def fmt2(idade):
    ''' Formata intervalo indefinido para um intervalo de idade válido '''
    if idade=='(8, 12)':
        return '(8, 13)'
    elif idade=='(38, 42)':
        return '(38, 43)'
    elif idade=='(27, 32)':
        return '(25, 32)'
    else:
        return None
    


def wrapper_fmt(idade):
    return fmt1(idade) if len(idade) < 3 else fmt2(idade)



def categorical_fmt(idade_range):
    if idade_range=='(0, 2)':
        return 0
    elif idade_range=='(4, 6)':
        return 1
    elif idade_range=='(8, 13)':
        return 2
    elif idade_range=='(15, 20)':
        return 3
    elif idade_range=='(25, 32)':
        return 4
    elif idade_range=='(38, 43)':
        return 5
    elif idade_range=='(48, 53)':
        return 6
    elif idade_range=='(60, 100)':
        return 7
   
    
def teste ():
    print("OK")
    
    
def load_data(mat_path):
    d = loadmat(mat_path)
   
    return d["image"], d["gender"][0], d["age"][0]
    #return d["image"], d["gender"][0], d["age"][0], d["db"][0], d["img_size"][0, 0], d["min_score"][0, 0]
    
    
    
def wrapper_data(name,data_name):
    imagens = []
    labels = []
    for data in data_name:
        obj = load(name+data)
        for x, y in zip(obj['imagens'], obj['labels']):
            print(x,'-',y)
            imagens.append(x)
            labels.append(y)
    return np.array(imagens, dtype=np.float16),  np.array(labels, dtype=np.uint16)
   
    


def get_imgpath(directory):
    for _,_, img_path in os.walk(directory):
        print("Quant Imgs:",len(img_path))
    return img_path        


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)        
