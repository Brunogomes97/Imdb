import cv2
import numpy as np
import matplotlib.pyplot as plt
from wide_resnet import WideResNet
from utils import get_meta

path_root = 'wiki_crop/'
cols, rows = 4, 3
img_num = cols * rows
model = WideResNet(64,16,8)()
model.load_weights('checkpoints/imdbase/weights.29-3.75.hdf5')

def prediction(img):
    img = cv2.resize(img,(64,64))
    img = np.expand_dims(np.array(img),axis = 0)
    result = model.predict(img)
    ages = np.arange(0, 101).reshape(101, 1)
    age_pred = result[1].dot(ages).flatten()
    gender_pred = result[0]
    if gender_pred[0][0] < 0.5:
        gender_pred = 'Homem'
    else:
        gender_pred = 'Mulher'    

    return str(gender_pred+'|'+str((int(age_pred))))
    
    
def show_imgs(img_paths):
  
    img_ids = np.random.choice(len(img_paths), img_num, replace=False)
    for i, img_id in enumerate(img_ids):
        print(img_id)
        plt.subplot(rows, cols, i + 1)
        img = cv2.imread(path_root + str(img_paths[img_id]))
        plt.title(str(prediction(img)))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    
    plt.show()



mat_path = str(path_root+'wiki.mat')


full_path, dob, gender, photo_taken, face_score, second_face_score, age\
    = get_meta(mat_path, 'wiki')
    
    
img_paths = []    

for i in range(len(face_score)):
    if face_score[i] >= 1.0 and np.isnan(second_face_score[i]):    
        img_paths.append(full_path[i][0])

print("#images with scores >= than 1.0 and no second face: {}".format(len(img_paths)))
                              
show_imgs(img_paths)      


