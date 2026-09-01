"""
llm_client.py
-------------
Wrapper γύρω από το Gemini API μέσω langchain-google-genai.

Γιατί το κάνουμε έτσι:
- Το LangGraph (που θα χρησιμοποιήσουμε για το agent orchestration) δουλεύει
  με οποιοδήποτε LangChain "chat model" - δεν είναι δεμένο με το OpenAI.
  Αντικαθιστούμε λοιπόν το OpenAI GPT με το Gemini (δωρεάν tier) απλά
  αλλάζοντας ΠΟΙΟ chat model φτιάχνουμε εδώ, χωρίς να αλλάξει τίποτα
  στη λογική του agent αργότερα.
- Η ChatGoogleGenerativeAI κλάση υποστηρίζει function/tool calling,
  που είναι απαραίτητο για τον agent μας (θα καλεί εργαλεία όπως
  Google Calendar, Gmail, κλπ).
- Χρησιμοποιούμε lazy singleton pattern (ίδια λογική με το redis_client)
  ώστε το μοντέλο να φτιάχνεται μία φορά, όχι σε κάθε κλήση.
"""

from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import get_settings


@lru_cache
def get_llm(temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    """
    Επιστρέφει ένα configured Gemini chat model, έτοιμο για χρήση από
    το LangGraph agent.

    Παράμετροι:
        temperature: πόσο "δημιουργικές"/τυχαίες θα είναι οι απαντήσεις.
                     Χαμηλό (0.0-0.3) = πιο προβλέψιμο, καλύτερο για
                     agents που καλούν εργαλεία (tool calling) όπου
                     θέλουμε συνέπεια, όχι δημιουργικότητα.

    Σημείωση: το @lru_cache εδώ σημαίνει ότι αν καλέσουμε get_llm(0.3)
    δύο φορές, θα πάρουμε το ΙΔΙΟ object (δεν ξαναφτιάχνεται) - αυτό
    εξοικονομεί λίγη μνήμη/χρόνο αρχικοποίησης. Αν καλέσουμε get_llm(0.7)
    θα φτιαχτεί ΝΕΟ object με άλλη temperature, και θα cache-αριστεί
    ξεχωριστά.
    """
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=temperature,
    )
