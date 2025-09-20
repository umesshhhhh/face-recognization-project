import cv2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") # Load Haar Cascade for face detection
video_cap = cv2.VideoCapture(0) # Open webcam
if not video_cap.isOpened():
    print("Error: Could not open camera")
    exit()
while True:
    ret, frame = video_cap.read()
    if not ret:
        break 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale (better for detection)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))     # Detect faces
    for (x, y, w, h) in faces:   # Draw rectangles around faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,0), 2)
    cv2.imshow("Face Detection", frame)     # Show the video
    if cv2.waitKey(10) & 0xFF == ord('x'):    # Press 'x' to exit
        break
video_cap.release()
cv2.destroyAllWindows()
