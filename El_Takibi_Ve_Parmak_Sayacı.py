#Projenin çalışması için opencv-pyhton ve mediapipe kütüphaneleri eklenmeli.
#Giray Canpolat
import cv2
import mediapipe as mp


mp_cizim = mp.solutions.drawing_utils
mp_eller = mp.solutions.hands

with mp_eller.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2) as eller:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()

        #BGR to RGB
        resim = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #Kamerayı dikey eksende aynalıyoruz (Sağ ve sol elin algılanmasının önemli olduğu projeler için, bu projede şart değil)
        resim = cv2.flip(resim, 1)

        #set flag
        resim.flags.writeable = False

        #Tespit
        sonuc = eller.process(resim)

        #set flag to true
        resim.flags.writeable = True

        #RGB to BGR
        resim = cv2.cvtColor(resim, cv2.COLOR_RGB2BGR)

        #tespit
        print(sonuc)

        parmakSayacı = 0

        #Sonucu render'layarak el çizimini oluşturuyoruz.
        if sonuc.multi_hand_landmarks:
            for num, el in enumerate(sonuc.multi_hand_landmarks):

                el_Index = sonuc.multi_hand_landmarks.index(el)
                el_Label = sonuc.multi_handedness[el_Index].classification[0].label
                el_isaretleri = []  # x ve y pozisyonlarını tutulması için.

                mp_cizim.draw_landmarks(resim, el, mp_eller.HAND_CONNECTIONS,
                                        mp_cizim.DrawingSpec(color=(139, 139, 122), thickness=2, circle_radius=4),
                                        mp_cizim.DrawingSpec(color=(255, 228, 19), thickness=2, circle_radius=2),
                                        )
                for landmarks in el.landmark:
                    el_isaretleri.append([landmarks.x, landmarks.y])

                if el_Label == "Left" and el_isaretleri[4][0] > el_isaretleri[3][0]:
                    parmakSayacı = parmakSayacı + 1
                elif el_Label == "Right" and el_isaretleri[4][0] < el_isaretleri[3][0]:
                    parmakSayacı = parmakSayacı + 1

                # İşaret parmağı, orta parmak, yüzük parmağı ve serçe parmağı için konumlarının
                # karşılaştırmalar yapılıyor ve buna göre sayacı arttırıyor.
                if el_isaretleri[8][1] < el_isaretleri[6][1]:  # İşaret parmağı
                    parmakSayacı = parmakSayacı + 1
                if el_isaretleri[12][1] < el_isaretleri[10][1]:  # Orta Parmak
                    parmakSayacı = parmakSayacı + 1
                if el_isaretleri[16][1] < el_isaretleri[14][1]:  # Yüzük parmağı
                    parmakSayacı = parmakSayacı + 1
                if el_isaretleri[20][1] < el_isaretleri[18][1]:  # Serçe parmağı
                    parmakSayacı = parmakSayacı + 1

        #Burada penceremize bir text ekleyerek "parmakSayacı" değişkenimizi yazdırıyoruz.
        cv2.putText(resim, str(parmakSayacı), (510, 450), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 3, (245, 10, 192), 10)

        cv2.imshow("El takip penceresi", resim)

        if cv2.waitKey(10) & 0xFF == ord("q"): #Programı kapatmak için klavyeden "q" tuşuna basmanız yeterli.
            break

cap.release()
cv2.destroyAllWindows()

mp_eller.HAND_CONNECTIONS

sonuc.multi_hand_landmarks