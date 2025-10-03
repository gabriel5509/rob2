import RPi.GPIO as GPIO
import board
import busio
from adafruit_pca9685 import PCA9685
import mediapipe as mp
from picamera2 import Picamera2
import cv2  # para manipulação de frames, sem imshow

# Sensor ultrassônico
TRIG = 27
ECHO = 17

GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

def get_distance():
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    pulse_start, pulse_end = None, None

    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()
    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 34300 / 2
    return distance


# PCA9685
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50

def set_servo_angle(channel, angle):
    pulse = int(150 + (angle / 180.0) * (600 - 150))
    pca.channels[channel].duty_cycle = pulse << 4

# Mediapipe
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.2)

# Picamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()
time.sleep(1)

frame_count = 0
processar_cada = 1  # processa 1 frame a cada N frames

# Loop principal
try:
    while True:
        dist = get_distance()
        print(f"Distância: {dist:.2f} cm")

        # Controle de servos baseado na distância
        if 10 < dist < 11:
            set_servo_angle(1, 90)
            time.sleep(2)
            set_servo_angle(1, 0)
        elif dist < 10:
            set_servo_angle(1, 90)
            time.sleep(2)
            set_servo_angle(1, 0)
            time.sleep(2)

        # Captura de frame RGB
        frame = picam2.capture_array()
        if frame.shape[-1] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        elif frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        frame_count += 1

        # Processa detecção de rosto apenas 1 a cada N frames
        if frame_count % processar_cada == 0:
            results = face_detection.process(frame)
            if results.detections:
                print("Rosto detectado!")
                # exemplo de ação com servo
                set_servo_angle(7, 90)
                time.sleep(3)
                set_servo_angle(7, 0)
                time.sleep(2)

        time.sleep(0.5)

except KeyboardInterrupt:

    print("Encerrando...")

finally:
    picam2.stop()
    pca.deinit()
    GPIO.cleanup()
