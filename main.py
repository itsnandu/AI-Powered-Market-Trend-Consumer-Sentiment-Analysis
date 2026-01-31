from notification.notification import *
# from .notification.notification import *


send_mail(text="Hello this is test user", subject="Testing mail")

send_slack_notification(text="Hello this is my second notification")

# import smtplib

# server = smtplib.SMTP("smtp.gmail.com", 587)
# server.starttls()

# server.login("nanduchoudhary1203@gmail.com", "kuck fdno love jewy")