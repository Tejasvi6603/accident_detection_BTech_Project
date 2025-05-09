import cv2
from detection import AccidentDetectionModel
import numpy as np
import pyttsx3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Email Configuration
EMAIL_SENDER = "tejuborole6603@gmail.com"  # Replace with your email
EMAIL_PASSWORD = "bilp zxoq qicz nbeo"  # Replace with your app-specific password (Not your actual password)
EMAIL_RECEIVER = "tejuborole6603@gmail.com"  # Replace with the recipient's email

# Initialize AI Voice Engine
engine = pyttsx3.init()

def speak_alert(probability):
    """Convert text to speech when an accident is detected"""
    alert_message = f"Warning! Accident detected with a probability of {probability}%. Stay safe."
    engine.say(alert_message)
    engine.runAndWait()

def send_email_alert(probability):
    """Send an emergency email when an accident is detected"""
    subject = "🚨 Accident Alert!"
    body = f"An accident has been detected with a probability of {probability}%. Please take necessary action."

    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()
        print("✅ Email alert sent successfully!")
    except Exception as e:
        print("❌ Failed to send email:", e)

# Load the accident detection model
model = AccidentDetectionModel("model.json", 'model_weights.h5')
font = cv2.FONT_HERSHEY_SIMPLEX

def startapplication():
    video = cv2.VideoCapture('cars.mp4')  # Use '0' for live webcam
    while True:
        ret, frame = video.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (250, 250))
        
        pred, prob = model.predict_accident(roi[np.newaxis, :, :])

        if pred == "Accident":
            prob = round(prob[0][0] * 100, 2)

            if prob > 90:
                speak_alert(prob)  # AI Voice Alert
                send_email_alert(prob)  # Email Notification

            cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, pred + " " + str(prob) + "%", (20, 30), font, 1, (255, 255, 0), 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

        cv2.imshow('Video', frame)

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    startapplication()
