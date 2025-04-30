import imaplib
import email
from email.header import decode_header
import re
import csv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Your email credentials (use an app password if you have 2FA enabled)
username = "analytech.clients@gmail.com"
password = "ejnv kdjr idks ygsx"

# Connect to Gmail's IMAP server
mail = imaplib.IMAP4_SSL("imap.gmail.com")
mail.login(username, password)

# Select the mailbox you want to scrape (INBOX, Sent, etc.)
mail.select("inbox")

# Search for all emails (you can modify this to filter by date, subject, etc.)
status, messages = mail.search(None, "ALL")  # Search for all emails

# Get the email IDs (they are returned as a space-separated list)
email_ids = messages[0].split()

# Function to clean email content (remove special characters, line breaks, and HTML tags)
def clean_email_content(text):
    # Remove HTML tags using regular expressions
    text = re.sub(r'<.*?>', '', text)  # Remove any HTML tags
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove excessive whitespaces
    text = re.sub(r'\s+', ' ', text)
    # Convert to lowercase
    text = text.lower()
    return text

# Function to analyze sentiment using VADER
def get_vader_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)['compound']
    if sentiment_score > 0.1:
        return 'positive'
    elif sentiment_score < -0.1:
        return 'negative'
    else:
        return 'neutral'

# List to store the email data (subject, body, and sentiment)
email_data = []

# Fetch all emails (may take longer for a large number of emails)
for email_id in email_ids:  # Loop through all email IDs
    status, msg_data = mail.fetch(email_id, "(RFC822)")
    for response_part in msg_data:
        if isinstance(response_part, tuple):
            msg = email.message_from_bytes(response_part[1])
            
            # Decode the email subject
            subject, encoding = decode_header(msg["Subject"])[0]
            if isinstance(subject, bytes):
                subject = subject.decode(encoding if encoding else "utf-8")

            # Get the email body (either text or HTML)
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))
                    
                    if content_type == "text/plain" and "attachment" not in content_disposition:
                        body = part.get_payload(decode=True).decode()
                        break
            else:
                body = msg.get_payload(decode=True).decode()

            # Clean the body
            cleaned_body = clean_email_content(body)

            # Get sentiment of the email
            sentiment = get_vader_sentiment(cleaned_body)

            # Append the email data (subject, cleaned body, and sentiment) to the list
            email_data.append([subject, cleaned_body, sentiment])

# Save data to CSV with Sentiment labels
with open("labeled_email_data.csv", "w", newline='', encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Subject", "Body", "Sentiment"])  # Header row with Sentiment
    writer.writerows(email_data)

print("Email data with sentiment labels has been saved to 'labeled_email_data.csv'.")

# Logout from the server
mail.logout()
