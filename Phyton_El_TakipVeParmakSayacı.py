import cv2
import mediapipe as mp
import numpy as np
import uuid
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=2) as hands:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()

        parmakSayacı = 0

        #BGR to RGB
        resim = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #set flag
        resim.flags.writeable = False

        #Tespit
        sonuc = hands.process(resim)

        #set flag to true
        resim.flags.writeable = True

        #RGB to BGR
        resim = cv2.cvtColor(resim, cv2.COLOR_RGB2BGR)

        #tespit
        print(sonuc)

        #Sonucu render'layarak el iskeletini oluşturuyoruz
        if sonuc.multi_hand_landmarks:
            for num, hand in enumerate(sonuc.multi_hand_landmarks):
                mp_drawing.draw_landmarks(resim, hand, mp_hands.HAND_CONNECTIONS,
                                                       mp_drawing.DrawingSpec(color=(139, 139, 122), thickness=2, circle_radius=4),
                                                       mp_drawing.DrawingSpec(color=(255, 228, 19), thickness=2, circle_radius=2),
                                            )

        #Burada pencerimeze bir text ekleyerek "parmakSayacı" değişkenimizi yazdırıyoruz.
        cv2.putText(resim, str(parmakSayacı), (510, 450), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 3, (245, 192, 192), 10)

        cv2.imshow("El takip penceresi", resim)

        if cv2.waitKey(10) & 0xFF == ord("q"): #Programı kapatmak için klavyeden "q" tuşuna basmanız yeterli.
            break





cap.release()
cv2.destroyAllWindows()

mp_hands.HAND_CONNECTIONS

sonuc.multi_hand_landmarks