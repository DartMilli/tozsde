import os
import smtplib
from email.mime.text import MIMEText
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


def send_email(subject, body, to_email):
    host = os.getenv("EMAIL_HOST")
    port = int(os.getenv("EMAIL_PORT", 587))
    user = os.getenv("EMAIL_USER")
    password = os.getenv('EMAIL_PASSWORD')

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_email

    try:
        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(msg)
        logger.info(f"{subject} e-mail elküldve")
        return True
    except Exception as e:
        logger.error(f"Email küldés hiba: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    send_email("teszt", "Teszt e-amil", "szlavikmilan@gmail.com")
