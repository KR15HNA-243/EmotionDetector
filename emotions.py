import cv2
import pandas as pd
from deepface import DeepFace
import matplotlib.pyplot as plt

emotion_frames = { 'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0 } #emotions available

#video_path = 'E:/Downloads/3695949-hd_1920_1080_24fps (1).mp4' #Video 1 of Smiling child
video_path = "E:/Downloads/5993541-hd_1920_1080_30fps (1).mp4" #Video 2 of Sad man
cap = cv2.VideoCapture(video_path)

#cap=cv2.VideoCapture(0) - To use the webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False) #using the DeepFace library to analyze emotions
        dominant_emotion = results[0]['dominant_emotion']
        emotion_frames[dominant_emotion] += 1

        cv2.putText(frame, f"Emotion: {dominant_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Emotion Detection', frame) #displaying the emotions at each frame

    except Exception as e:
        print(f"Error: {e}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() #closing the opencv window

most_emotion = max(emotion_frames, key=emotion_frames.get)
frame_count = emotion_frames[most_emotion]
print(f"The most experienced emotion was {most_emotion} for {frame_count} frames.") #most experienced emotion analysis

emotion_df = pd.DataFrame(list(emotion_frames.items()), columns=['Emotion', 'Frame Count'])

emotion_df.plot(kind='bar', x='Emotion', y='Frame Count', legend=False)
plt.xlabel('Emotion')
plt.ylabel('Number of Frames')
plt.title('Emotion v/s Frame Count Summary')
plt.show() #visual representation of the emotions experienced in the video
