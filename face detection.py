import cv2

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open webcam
video_cap = cv2.VideoCapture(0)

if not video_cap.isOpened():
    print("Error: Could not open camera")
    exit()

while True:
    ret, frame = video_cap.read()
    if not ret:
        break

    # Convert to grayscale (better for detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,0), 2)

    # Show the video
    cv2.imshow("Face Detection", frame)

    # Press 'x' to exit
    if cv2.waitKey(10) & 0xFF == ord('x'):
        break

video_cap.release()
cv2.destroyAllWindows()
