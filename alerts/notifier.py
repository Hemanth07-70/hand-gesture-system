import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import cv2
import os
import time

class AlertEngine:
    def __init__(self, smtp_server="smtp.gmail.com", smtp_port=465, sender_email=None, receiver_email=None, password=None):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.receiver_email = receiver_email
        self.password = password
        
    def get_location(self):
        # Mock location - in a real app, use requests to ipinfo.io or similar
        return "Lat: 12.9716, Lon: 77.5946 (Bangalore, India)"

    def send_email_alert(self, frame, message):
        """Send email with message and captured frame."""
        if not all([self.sender_email, self.receiver_email, self.password]):
            print(f"DEBUG: Email alert not configured. Message: {message}")
            return False

        try:
            msg = MIMEMultipart()
            msg['Subject'] = "🆘 DISTRESS SIGNAL DETECTED"
            msg['From'] = self.sender_email
            msg['To'] = self.receiver_email

            location = self.get_location()
            body = f"{message}\n\nTime: {time.ctime()}\nLocation: {location}"
            msg.attach(MIMEText(body, 'plain'))

            # Attach Frame
            _, img_encoded = cv2.imencode('.jpg', frame)
            image_attachment = MIMEImage(img_encoded.tobytes(), name="distress_frame.jpg")
            msg.attach(image_attachment)

            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, context=context) as server:
                server.login(self.sender_email, self.password)
                server.send_message(msg)
            
            print(f"SUCCESS: Email alert sent to {self.receiver_email}")
            return True
        except Exception as e:
            print(f"ERROR: Failed to send email alert: {e}")
            return False

    def send_local_notification(self, message):
        """Display a system notification (macOS specific)."""
        os.system(f'osascript -e \'display notification "{message}" with title "Distress Alert"\'')

    def trigger(self, frame, message):
        print(f"TRIGGERED: {message}")
        self.send_local_notification(message)
        # Email is sent only if configured
        self.send_email_alert(frame, message)
