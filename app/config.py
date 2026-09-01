"""
config.py
---------
Κεντρικοποιημένη διαχείριση ρυθμίσεων του project.

Γιατί το κάνουμε έτσι:
- Αντί να διαβάζουμε os.environ["..."] σκόρπια σε 10 διαφορετικά αρχεία
  (κάτι που είναι εύκολο να ξεχάσουμε ή να γράψουμε λάθος το όνομα),
  έχουμε ΕΝΑ Settings object που το κάνουμε import παντού.
- Το pydantic-settings κάνει αυτόματα validation: αν λείπει μια τιμή
  που έχουμε ορίσει ως υποχρεωτική, το πρόγραμμα θα σκάσει ΚΑΤΑ ΤΗΝ
  ΕΚΚΙΝΗΣΗ με σαφές μήνυμα, αντί να σκάσει αργότερα μέσα σε ένα request
  με ένα confusing KeyError.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Gemini / LLM ---
    google_api_key: str
    gemini_model: str = "gemini-3.6-flash"

    # --- Redis ---
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # --- Google OAuth (Calendar / Gmail) ---
    google_oauth_client_id: str = ""
    google_oauth_client_secret: str = ""
    google_oauth_redirect_uri: str = "http://localhost:8000/auth/callback"

    # --- App ---
    app_env: str = "development"
    log_level: str = "INFO"

    # --- Πρωινή ενημέρωση ---
    # briefing_enabled: αν θα τρέχει αυτόματα στην προγραμματισμένη ώρα.
    #     Ακόμα κι αν είναι False, μπορείς να τη ζητήσεις χειροκίνητα από
    #     το UI ή το endpoint /briefing.
    # briefing_send_email: αν θα σου τη στέλνει και με email (στη δική σου
    #     διεύθυνση, που διαβάζεται από το προφίλ του λογαριασμού).
    briefing_enabled: bool = True
    briefing_hour: int = 8
    briefing_minute: int = 30
    briefing_send_email: bool = True

    # Λέει στο pydantic-settings να διαβάσει τιμές από το αρχείο .env
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # αγνόησε τυχόν επιπλέον μεταβλητές στο .env
    )


@lru_cache
def get_settings() -> Settings:
    """
    Επιστρέφει ένα singleton instance των Settings.

    Το @lru_cache σημαίνει ότι το .env διαβάζεται/παρσάρεται ΜΙΑ φορά,
    όχι σε κάθε request — μικρή αλλά σωστή βελτιστοποίηση.
    """
    return Settings()
