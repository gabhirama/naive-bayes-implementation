import os.path
import base64
from typing import Optional
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def authenticate_gmail():
    """
    Authenticates the user with the Gmail API using OAuth 2.0.
    Caches credentials in token.json to avoid re-logging in.
    Returns the authorized Gmail API service object.
    """
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=8080)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    try:
        service = build('gmail', 'v1', credentials=creds)
        return service
    except HttpError as error:
        print(f'An error occurred: {error}')
        return None

def get_messages(service, query='', max_results=100) -> list[dict]:
    """Lists the user's messages that match the query."""
    try:
        result = service.users().messages().list(userId='me', q=query, maxResults=max_results).execute()
        return result.get('messages', [])
    except HttpError as error:
        print(f'An error occurred while fetching messages: {error}')
        return []

def get_email_details(service, msg_id) -> Optional[dict]:
    """
    Gets the full details of a single email in one API call.
    Returns a dictionary with subject, sender, date, and content.
    """
    try:
        msg = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
        payload = msg.get('payload', {})
        headers = payload.get('headers', [])
        
        subject = next((d['value'] for d in headers if d['name'].lower() == 'subject'), '')
        sender = next((d['value'] for d in headers if d['name'].lower() == 'from'), '')
        date = next((d['value'] for d in headers if d['name'].lower() == 'date'), '')
        
        body = ''
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain' and 'data' in part['body']:
                    body += base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', 'ignore')
                    break # Stop after finding the first plain text part
        elif 'data' in payload.get('body', {}):
            body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', 'ignore')
            
        return {'subject': subject, 'sender': sender, 'date': date, 'content': body}
    except HttpError as error:
        print(f'An error occurred for message ID {msg_id}: {error}')
        return None

def fetch_spam_and_ham(max_results_per_category: int) -> tuple[dict, dict]:
    """
    Main function to authenticate, fetch, and process spam and ham emails.
    Returns two dictionaries: one for spam and one for ham.
    """
    service = authenticate_gmail()
    if not service:
        return {}, {}

    spam_messages = get_messages(service, query='label:SPAM', max_results=max_results_per_category)
    ham_messages = get_messages(service, query='category:primary -label:SPAM', max_results=max_results_per_category)
    
    print(f"Found {len(spam_messages)} spam messages and {len(ham_messages)} ham messages.")
    
    spam_content = {}
    for msg in spam_messages:
        details = get_email_details(service, msg['id'])
        if details:
            spam_content[msg['id']] = details
            
    ham_content = {}
    for msg in ham_messages:
        details = get_email_details(service, msg['id'])
        if details:
            ham_content[msg['id']] = details
            
    return spam_content, ham_content

# if __name__ == '__main__':
#     # This block runs if you execute this script directly (e.g., `python gmail_api.py`)
#     # It will fetch the emails and save them to JSON files, just like your notebook.
#     spam_data, ham_data = fetch_spam_and_ham(100)
#     import json
#     with open('spam_content.json', 'w', encoding='utf-8') as f:
#         json.dump(spam_data, f, indent=4, ensure_ascii=False)
#     with open('ham_content.json', 'w', encoding='utf-8') as f:
#         json.dump(ham_data, f, indent=4, ensure_ascii=False)
        
#     print("\nProcess complete. Spam and ham content saved to JSON files.")