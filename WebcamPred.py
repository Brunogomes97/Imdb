import cv2
import dlib
import numpy as np
from contextlib import contextmanager
from wide_resnet import WideResNet
from functions import draw_label ##Desenhar rotulo na face

"""
Captura de Video por gerador yield
"""
@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def yield_images():
    # CAptura de Video
    with video_capture(0) as cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

        while True:
            # Frame
            ret, img = cap.read()

            if not ret:
                raise RuntimeError("Failed to capture image")

            yield img


#########################################################

profundidade = 16
k = 8
pesos = 'checkpoints/imdbase/weights.29-3.75.hdf5'
margem = 0.4
img_size = 64


detector = dlib.get_frontal_face_detector()

model = WideResNet(img_size,profundidade,k)()
#model.summary()
model.load_weights(pesos)

image_generator = yield_images()
print("ok2")
for img in image_generator:
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)

        # detect faces using dlib detector
        detected = detector(input_img, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))


        ## Retangulo OpenCV
        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - margem * w), 0)
                yw1 = max(int(y1 - margem * h), 0)
                xw2 = min(int(x2 + margem * w), img_w - 1)
                yw2 = min(int(y2 + margem * h), img_h - 1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

            # predict Idade e Genero
            results = model.predict(faces)
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()

            # Plotar resultos no Video
            for i, d in enumerate(detected):
                label = "{}, {}".format(int(predicted_ages[i]),
                                        "M" if predicted_genders[i][0] < 0.5 else "F")
                draw_label(img, (d.left(), d.top()), label)

        cv2.imshow("result", img)
        
        
        key = cv2.waitKey(30)

        if key == 27:  # ESC
            break




