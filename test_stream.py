import cv2

url = "http://10.155.155.37:81/stream"

cap = cv2.VideoCapture(url)

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed")
        break
        
    cv2.imshow("ESP32 Stream", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()



