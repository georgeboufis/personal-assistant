"""
memory.py
---------
Διαχείριση conversation memory (ιστορικού συνομιλίας) στο Redis.

Γιατί το χρειαζόμαστε:
Το LangGraph agent μας (graph.py) δεν έχει καμία μνήμη από μόνο του -
κάθε φορά που καλούμε agent.invoke(), ξεκινάει από το state που ΤΟΥ
δίνουμε εμείς. Αν θέλουμε ο χρήστης να μπορεί να πει "συνέχισε από εκεί
που μείναμε" σε ένα δεύτερο HTTP request, πρέπει ΕΜΕΙΣ να αποθηκεύσουμε
το ιστορικό κάπου ανάμεσα στα requests - εδώ μπαίνει το Redis.

Γιατί χρειαζόμαστε (de)serialization:
Τα LangChain message objects (HumanMessage, AIMessage) είναι Python
objects, όχι strings. Το Redis όμως αποθηκεύει μόνο strings/bytes.
Άρα πρέπει να τα μετατρέψουμε σε κάτι που μπορεί να γραφτεί σαν string
(JSON) πριν τα αποθηκεύσουμε, και να τα ξαναφτιάξουμε σε message objects
όταν τα διαβάζουμε.

Σχεδιαστική απόφαση για τη μνήμη του Mac (8GB):
Βάζουμε TTL (time-to-live) σε κάθε session ώστε παλιές συνομιλίες να
σβήνονται αυτόματα μετά από κάποιο διάστημα, αντί να συσσωρεύονται στο
Redis επ' αόριστον και να μεγαλώνει η κατανάλωση μνήμης του Redis server
χωρίς όριο.
"""

import json
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
from app.core import redis_client

# Πόσο καιρό κρατάμε μια συνομιλία στο Redis πριν σβηστεί αυτόματα.
# 24 ώρες είναι λογικό default για έναν προσωπικό assistant.
SESSION_TTL_SECONDS = 60 * 60 * 24

# Πρόθεμα (prefix) σε όλα τα Redis keys που αφορούν conversation history,
# ώστε να ξεχωρίζουν εύκολα από άλλα δεδομένα που ίσως βάλουμε αργότερα
# στο ίδιο Redis instance (π.χ. καθέ κλειδί θα είναι "conversation:abc123").
KEY_PREFIX = "conversation:"


def _key_for(session_id: str) -> str:
    """Χτίζει το Redis key για ένα δεδομένο session_id."""
    return f"{KEY_PREFIX}{session_id}"


def load_history(session_id: str) -> list[BaseMessage]:
    """
    Φορτώνει το ιστορικό μηνυμάτων ενός session από το Redis.

    Αν δεν υπάρχει τίποτα αποθηκευμένο (νέο session, ή έληξε το TTL),
    επιστρέφει κενή λίστα - ο agent απλά ξεκινάει από το μηδέν, χωρίς
    να σκάσει.
    """
    client = redis_client.get_redis_client()
    raw = client.get(_key_for(session_id))

    if raw is None:
        return []

    as_dicts = json.loads(raw)
    return messages_from_dict(as_dicts)


def save_history(session_id: str, messages: list[BaseMessage]) -> None:
    """
    Αποθηκεύει το (ενημερωμένο) ιστορικό μηνυμάτων ενός session στο Redis,
    με TTL ώστε να σβήνεται αυτόματα αν δεν χρησιμοποιηθεί ξανά.
    """
    client = redis_client.get_redis_client()
    as_dicts = messages_to_dict(messages)
    client.set(
        _key_for(session_id),
        json.dumps(as_dicts, ensure_ascii=False),
        ex=SESSION_TTL_SECONDS,
    )


def clear_history(session_id: str) -> None:
    """Διαγράφει το ιστορικό ενός session (π.χ. αν ο χρήστης πει 'ξέχασε τα πάντα')."""
    client = redis_client.get_redis_client()
    client.delete(_key_for(session_id))
