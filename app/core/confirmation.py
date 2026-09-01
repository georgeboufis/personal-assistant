"""
confirmation.py
----------------
Η λογική του "confirmation layer": πώς καταλαβαίνουμε ότι μια ενέργεια
περιμένει έγκριση, και πώς ερμηνεύουμε την απάντηση του χρήστη.

Η κεντρική ιδέα - γιατί ΔΕΝ χρειαζόμαστε ξεχωριστή αποθήκευση:
Όταν ο agent ζητάει επικίνδυνο εργαλείο, το graph σταματάει αφήνοντας ένα
AIMessage με tool_calls ΧΩΡΙΣ αντίστοιχα ToolMessages. Αυτή η "τρύπα" στο
ιστορικό ΕΙΝΑΙ η ένδειξη ότι κάτι εκκρεμεί. Αφού το ιστορικό ήδη σώζεται
στο Redis, δεν χρειάζεται δεύτερο σύστημα αποθήκευσης - μία πηγή αλήθειας.
"""

import unicodedata

from langchain_core.messages import AIMessage, ToolMessage

# Λέξεις που θεωρούμε ξεκάθαρη έγκριση / άρνηση, αν ο χρήστης γράψει
# ελεύθερο κείμενο αντί να πατήσει τα κουμπιά του UI.
# Είναι γραμμένες ΧΩΡΙΣ τόνους, γιατί τους αφαιρούμε πριν τη σύγκριση
# (ώστε "ναί", "ναι", "ΝΑΙ" να μετράνε όλα το ίδιο).
AFFIRMATIVE = {
    "ναι", "ναι παρακαλω", "οκ", "ενταξει", "προχωρα", "προχωρησε",
    "κανε το", "στειλε", "στειλ το", "στειλε το", "εγκρινω", "συμφωνω",
    "yes", "y", "ok", "okay", "sure", "go", "go ahead", "do it",
    "send", "send it", "confirm", "approve",
}

NEGATIVE = {
    "οχι", "οχι ευχαριστω", "ακυρο", "ακυρωση", "ακυρωσε", "μην",
    "μην το κανεις", "σταματα", "αρνουμαι", "no", "n", "nope",
    "cancel", "stop", "abort", "dont", "don t", "do not",
}


def _normalize(text: str) -> str:
    """
    Κανονικοποιεί κείμενο για σύγκριση: πεζά, χωρίς τόνους, χωρίς σημεία
    στίξης στην αρχή/τέλος.

    Το unicodedata.normalize("NFD") "σπάει" τους τονισμένους χαρακτήρες σε
    γράμμα + τόνο, και μετά πετάμε τα σημάδια τόνου (category "Mn").
    Έτσι το "ναί" γίνεται "ναι".
    """
    text = text.strip().lower()
    text = "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )
    return text.strip(" .,!;:·?\n\t")


def get_pending_tool_calls(messages: list) -> list[dict]:
    """
    Επιστρέφει τα tool_calls που περιμένουν έγκριση, ή κενή λίστα.

    Πώς το εντοπίζουμε: το ΤΕΛΕΥΤΑΙΟ μήνυμα είναι AIMessage με tool_calls.
    Αν είχαν εκτελεστεί, θα ακολουθούσαν ToolMessages - άρα το γεγονός ότι
    το AIMessage είναι τελευταίο σημαίνει ότι σταματήσαμε επίτηδες.
    """
    if not messages:
        return []

    last = messages[-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return last.tool_calls

    return []


def interpret_answer(message: str, confirm_flag: bool | None) -> bool | None:
    """
    Ερμηνεύει την απάντηση του χρήστη σε ένα αίτημα έγκρισης.

    Επιστρέφει:
        True  -> ο χρήστης ενέκρινε
        False -> ο χρήστης αρνήθηκε
        None  -> δεν απάντησε στην ερώτηση (έγραψε κάτι άσχετο)

    Προτεραιότητα στο confirm_flag: αν ο χρήστης πάτησε τα κουμπιά του UI,
    η πρόθεσή του είναι σαφής και δεν χρειάζεται να μαντέψουμε από κείμενο.
    """
    if confirm_flag is not None:
        return confirm_flag

    normalized = _normalize(message)

    if normalized in AFFIRMATIVE:
        return True
    if normalized in NEGATIVE:
        return False

    # Ο χρήστης έγραψε κάτι άλλο (π.χ. άλλαξε θέμα). ΔΕΝ το θεωρούμε
    # έγκριση - στην αμφιβολία, δεν εκτελούμε.
    return None


def build_decline_messages(
    tool_calls: list[dict], reason: str
) -> list[ToolMessage]:
    """
    Φτιάχνει ToolMessages που δηλώνουν ότι η ενέργεια ΔΕΝ εκτελέστηκε.

    Γιατί είναι απαραίτητο (και όχι απλά "ωραίο"):
    Τα μοντέλα απαιτούν κάθε tool_call να έχει αντίστοιχο αποτέλεσμα. Αν
    αφήναμε το AIMessage με tool_calls "ορφανό" στο ιστορικό και στέλναμε
    ξανά τη συνομιλία, το Gemini θα επέστρεφε σφάλμα. Άρα ακόμα και η
    άρνηση πρέπει να καταγραφεί ως αποτέλεσμα.
    """
    return [
        ToolMessage(content=reason, tool_call_id=tc["id"]) for tc in tool_calls
    ]
