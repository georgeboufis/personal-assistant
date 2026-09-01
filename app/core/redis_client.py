"""
redis_client.py
----------------
Wrapper γύρω από το redis-py client.

Γιατί το κάνουμε έτσι:
- Θέλουμε ΜΙΑ σύνδεση (connection pool) που μοιράζεται όλο το app,
  όχι μια καινούργια σύνδεση σε κάθε request (σπατάλη πόρων).
- Απομονώνουμε τη λογική του Redis εδώ, ώστε αν αργότερα αλλάξουμε
  provider (π.χ. σε Redis Cloud free tier) να αλλάξει μόνο αυτό το αρχείο.

Σημείωση για τη μνήμη του Mac (8GB):
- Το redis-py client αυτό καθαυτό είναι πανάλαφρο (μόνο connection pool,
  δεν κρατάει data στη μνήμη του Python process).
- Η ΠΡΑΓΜΑΤΙΚΗ κατανάλωση μνήμης γίνεται από τον Redis SERVER, που τρέχει
  σαν ξεχωριστό process. Θα το ρυθμίσουμε αργότερα με maxmemory cap
  (π.χ. 100-200MB) ώστε να μην "τρώει" απρόβλεπτα RAM ενώ τρέχει και
  το RAG project σου.
"""

import redis
from app.config import get_settings


_redis_client: redis.Redis | None = None


def get_redis_client() -> redis.Redis:
    """
    Επιστρέφει ένα (singleton) Redis client instance.

    Χρησιμοποιούμε lazy initialization: η σύνδεση δεν ανοίγει όταν γίνεται
    import το module, αλλά μόνο την πρώτη φορά που ζητηθεί - έτσι το app
    μπορεί να ξεκινήσει (π.χ. για testing) ακόμα και αν το Redis δεν τρέχει
    ακόμα, και θα σκάσει μόνο όταν όντως προσπαθήσουμε να το χρησιμοποιήσουμε.
    """
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True,  # επιστρέφει strings αντί για bytes
        )
    return _redis_client


def check_redis_connection() -> bool:
    """
    Ελέγχει αν η σύνδεση με το Redis είναι ενεργή.
    Χρησιμοποιείται στο health-check endpoint του FastAPI.
    """
    try:
        client = get_redis_client()
        return client.ping()
    except redis.exceptions.RedisError:
        return False
