"""
google_auth.py
---------------
Διαχείριση OAuth 2.0 authentication με τον προσωπικό σου Google λογαριασμό,
ώστε ο agent να μπορεί να διαβάζει/γράφει στο Calendar (και αργότερα Gmail)
ΕΚ ΜΕΡΟΥΣ σου.

Πώς δουλεύει το OAuth flow (σε απλά λόγια):
1. Την ΠΡΩΤΗ φορά, δεν έχουμε "άδεια" - ανοίγει ένα παράθυρο browser όπου
   ο ΙΔΙΟΣ ο χρήστης (εσύ) βλέπει μια οθόνη Google και πατάει "Allow".
2. Η Google μας δίνει πίσω ένα "token" (σαν ένα κλειδί με ημερομηνία λήξης).
3. Αποθηκεύουμε αυτό το token τοπικά (token.json) ώστε τις ΕΠΟΜΕΝΕΣ φορές
   να ΜΗΝ χρειάζεται να ξαναπερνάς από browser - το πρόγραμμα το διαβάζει
   κατευθείαν από το δίσκο.
4. Αν το token λήξει, το βιβλιοθήκη το ανανεώνει αυτόματα ("refresh"),
   χωρίς να χρειαστεί να περάσεις ξανά από browser - ΕΚΤΟΣ αν ανακαλέσεις
   εσύ χειροκίνητα την πρόσβαση από το Google Account σου.

Γιατί credentials.json ΚΑΙ token.json (δύο διαφορετικά αρχεία):
- credentials.json: ταυτοποιεί ΤΗΝ ΕΦΑΡΜΟΓΗ ("ποιο app ζητάει πρόσβαση") -
  το κατέβασες από το Google Cloud Console, είναι σταθερό.
- token.json: ταυτοποιεί ΤΗΝ ΑΔΕΙΑ ΣΟΥ ("ο Γιώργος επέτρεψε σε αυτό το app
  να διαβάζει το calendar του") - δημιουργείται αυτόματα την πρώτη φορά,
  προσωπικό, ΔΕΝ πρέπει ποτέ να μοιραστεί.
"""

import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# Τα "scopes" καθορίζουν ΤΙ ΑΚΡΙΒΩΣ επιτρέπουμε στο app να κάνει.
# Καλή πρακτική ασφαλείας: ζητάμε μόνο ό,τι χρειαζόμαστε, τίποτα παραπάνω.
#
# - calendar.events: read + write events (ΟΧΙ διαχείριση calendars/settings)
# - gmail.readonly:  ΜΟΝΟ ανάγνωση emails (δεν επιτρέπει καμία αλλαγή)
# - gmail.compose:   δημιουργία/επεξεργασία προχείρων ΚΑΙ αποστολή
# - tasks:           read + write στη λίστα εργασιών
#
# ΣΗΜΕΙΩΣΗ: δεν ζητάμε gmail.modify (θα επέτρεπε διαγραφές/αλλαγές labels)
# ούτε το πλήρες mail.google.com scope - δεν τα χρειαζόμαστε.
SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/tasks",
]

# Πού βρίσκονται τα δύο αρχεία - στο ROOT του project, όχι μέσα στο app/,
# ώστε να είναι εύκολο να τα εντοπίσει ο χρήστης και να τα διαγράψει αν
# χρειαστεί (π.χ. για να ανακαλέσει πρόσβαση).
CREDENTIALS_PATH = "credentials.json"
TOKEN_PATH = "token.json"


def get_google_credentials() -> Credentials:
    """
    Επιστρέφει έγκυρα Google credentials (για Calendar ΚΑΙ Gmail), κάνοντας
    ό,τι χρειάζεται για να τα αποκτήσει:
      - Αν υπάρχει ήδη έγκυρο token.json -> το χρησιμοποιεί κατευθείαν.
      - Αν υπάρχει αλλά έληξε -> το ανανεώνει αυτόματα (refresh).
      - Αν δεν υπάρχει καθόλου -> ανοίγει browser για πρώτη σύνδεση,
        μετά αποθηκεύει το αποτέλεσμα σε token.json για την επόμενη φορά.
    """
    creds = None

    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # Το token υπάρχει αλλά έληξε - ανανέωσέ το ΧΩΡΙΣ να χρειαστεί
            # να ξαναπεράσουμε από browser.
            creds.refresh(Request())
        else:
            # Δεν υπάρχει κανένα προηγούμενο token - πρώτη φορά σύνδεσης.
            if not os.path.exists(CREDENTIALS_PATH):
                raise FileNotFoundError(
                    f"Δεν βρέθηκε το {CREDENTIALS_PATH}. Κατέβασέ το από το "
                    "Google Cloud Console (Credentials -> OAuth Client ID) "
                    "και βάλε το στο root του project."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_PATH, SCOPES
            )
            # run_local_server ανοίγει έναν τοπικό, προσωρινό web server
            # (μόνο στο δικό σου Mac) για να "πιάσει" το redirect από τη
            # Google αφού πατήσεις "Allow" στον browser.
            creds = flow.run_local_server(port=0)

        # Αποθήκευσε το (νέο ή ανανεωμένο) token για την επόμενη φορά.
        with open(TOKEN_PATH, "w") as token_file:
            token_file.write(creds.to_json())

    return creds
