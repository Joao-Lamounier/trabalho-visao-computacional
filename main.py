import os

import cv2
import dlib
import numpy as np
import pygame
from scipy.spatial import distance


class DrowsinessDetector:
    def __init__(self):
        # Inicializa o detector facial do dlib
        self.face_detector = dlib.get_frontal_face_detector()

        # Carrega o preditor de pontos faciais do dlib
        current_dir = os.path.dirname(os.path.abspath(__file__))
        predictor_path = os.path.join(current_dir, "shape_predictor_68_face_landmarks.dat")
        self.landmark_predictor = dlib.shape_predictor(predictor_path)

        # Carrega o detector de olhos Haar Cascade
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Constantes para cálculo do EAR (Eye Aspect Ratio)
        # Índices dos pontos dos olhos no modelo facial do dlib
        self.LEFT_EYE_POINTS = list(range(36, 42))
        self.RIGHT_EYE_POINTS = list(range(42, 48))

        # Parâmetros para detecção de sonolência
        self.EAR_THRESHOLD = 0.25
        self.CONSECUTIVE_FRAMES = 20
        self.counter = 0

        # Configurações de alarme
        self.alarm_on = False
        pygame.mixer.init()
        self.alert_sound = pygame.mixer.Sound("alert.mp3")

    def calculate_ear(self, eye_points):
        """
        Calcula o Eye Aspect Ratio (EAR) para um olho
        EAR = (||p2-p6|| + ||p3-p5||) / (2||p1-p4||)
        """
        # Calcula distâncias verticais
        A = distance.euclidean(eye_points[1], eye_points[5])
        B = distance.euclidean(eye_points[2], eye_points[4])

        # Calcula distância horizontal
        C = distance.euclidean(eye_points[0], eye_points[3])

        # Calcula EAR
        ear = (A + B) / (2.0 * C)
        return ear

    def get_eye_points(self, facial_landmarks, eye_indices):
        """Extrai os pontos do olho dos landmarks faciais"""
        points = []
        for i in eye_indices:
            point = facial_landmarks.part(i)
            points.append([point.x, point.y])
        return np.array(points)

    def detect_eyes_haar(self, gray, face):
        """Detecta olhos usando Haar Cascade"""
        x, y, w, h = face
        roi_gray = gray[y:y + h, x:x + w]
        eyes = self.eye_cascade.detectMultiScale(roi_gray)
        return eyes, roi_gray

    def draw_eyes(self, frame, eyes, face_x, face_y):
        """Desenha retângulos ao redor dos olhos detectados"""
        for (ex, ey, ew, eh) in eyes:
            # Converte coordenadas relativas à face para coordenadas na imagem
            abs_x = face_x + ex
            abs_y = face_y + ey
            cv2.rectangle(frame, (abs_x, abs_y), (abs_x + ew, abs_y + eh), (0, 255, 0), 2)

    def process_frame(self, frame):
        # Converte para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Aplica CLAHE para melhorar o contraste local
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Aplica um filtro bilateral para suavizar a iluminação sem perder bordas
        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        # Aplica correção de gamma para melhorar detalhes em áreas muito escuras ou muito claras
        def adjust_gamma(image, gamma=1.2):
            inv_gamma = 1.0 / gamma
            table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(image, table)

        gray = adjust_gamma(gray, gamma=1.2)

        # Aplica top-hat e black-hat para remover sombras e melhorar detalhes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        gray = cv2.add(gray, tophat)  # Melhora áreas escuras
        gray = cv2.subtract(gray, blackhat)  # Reduz brilho excessivo

        return gray

    def detect_drowsiness(self, frame):
        # Converte para escala de cinza
        gray = self.process_frame(frame)

        # Detecta faces usando dlib
        faces = self.face_detector(gray)

        # Para cada face detectada
        for face in faces:
            # Obtém os landmarks faciais
            landmarks = self.landmark_predictor(gray, face)

            # Extrai pontos dos olhos
            left_eye_points = self.get_eye_points(landmarks, self.LEFT_EYE_POINTS)
            right_eye_points = self.get_eye_points(landmarks, self.RIGHT_EYE_POINTS)

            # Calcula EAR para ambos os olhos
            left_ear = self.calculate_ear(left_eye_points)
            right_ear = self.calculate_ear(right_eye_points)

            # Média do EAR
            avg_ear = (left_ear + right_ear) / 2.0

            # Coordenadas da face para o Haar Cascade
            x = face.left()
            y = face.top()
            w = face.right() - face.left()
            h = face.bottom() - face.top()

            # Detecta olhos usando Haar Cascade
            eyes, _ = self.detect_eyes_haar(gray, (x, y, w, h))

            # Desenha os olhos detectados
            self.draw_eyes(frame, eyes, x, y)

            # Verifica sonolência
            if avg_ear < self.EAR_THRESHOLD:
                self.counter += 1
                if self.counter >= self.CONSECUTIVE_FRAMES:
                    if not self.alarm_on:
                        self.alarm_on = True
                        self.alert_sound.play()

                    cv2.putText(frame, "ALERTA SONOLENCIA!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.counter = 0
                self.alarm_on = False
                self.alert_sound.stop()

            # Desenha o valor do EAR na tela
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Desenha retângulo ao redor da face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Desenha os pontos dos olhos
            for points in [left_eye_points, right_eye_points]:
                for point in points:
                    cv2.circle(frame, tuple(map(int, point)), 2, (0, 0, 255), -1)

        return frame


def main():
    # Verifica se o arquivo do preditor existe
    if not os.path.isfile("shape_predictor_68_face_landmarks.dat"):
        print("ERRO: Arquivo 'shape_predictor_68_face_landmarks.dat' não encontrado!")
        print("Por favor, baixe o arquivo em:")
        print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("Descompacte e coloque no mesmo diretório deste script.")
        return

    print("Iniciando detector de sonolência...")
    detector = DrowsinessDetector()

    print("Abrindo câmera...")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detector.detect_drowsiness(frame)

        cv2.imshow("Detector de Sonolencia", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()