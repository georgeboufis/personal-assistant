# Tests

## Εκτέλεση

```bash
# Όλα τα tests
pytest

# Ένα αρχείο
pytest tests/test_graph.py

# Μία κλάση ή ένα test
pytest tests/test_api.py::TestConfirmationFlow
pytest tests/test_api.py::TestConfirmationFlow::test_2_έγκριση_εκτελεί_ακριβώς_μία_φορά

# Με λεπτομέρειες όταν κάτι σπάει
pytest -v

# Σταμάτα στο πρώτο σφάλμα
pytest -x

# Κάλυψη κώδικα
pytest --cov=app --cov-report=term-missing
```

## Τι καλύπτει κάθε αρχείο

| Αρχείο | Περιεχόμενο |
|---|---|
| `conftest.py` | Κοινές fixtures (fake Redis, fake LLM, fake Google APIs) |
| `test_core.py` | Ρυθμίσεις, αποθήκευση ιστορικού, λογική έγκρισης |
| `test_calendar_tool.py` | Ζώνες ώρας, ανάγνωση και δημιουργία events |
| `test_gmail_tool.py` | MIME parsing, threading απαντήσεων, πρόχειρα vs αποστολή |
| `test_graph.py` | Δρομολόγηση agent, confirmation layer, χειρισμός σφαλμάτων |
| `test_api.py` | HTTP endpoints, πλήρεις διαδρομές έγκρισης |
| `test_briefing.py` | Πρωινή ενημέρωση, scheduler |

## Αρχές

**Καμία πραγματική κλήση σε API.** Τα tests δεν χτυπάνε ποτέ Google ή
Gemini, ούτε πραγματικό Redis. Έτσι τρέχουν σε δευτερόλεπτα, δεν
αποτυγχάνουν επειδή έπεσε το δίκτυο, και δεν υπάρχει περίπτωση να
στείλουν πραγματικό email.

**Τα κρίσιμα tests αφορούν ασφάλεια.** Αν πρόκειται να αλλάξεις κάτι,
αυτά είναι που δεν πρέπει ποτέ να σπάσουν:

- `test_graph.py::TestAgentFlow::test_επικίνδυνο_εργαλείο_ΔΕΝ_εκτελείται`
- `test_briefing.py::TestGenerateBriefing::test_το_μοντέλο_ΔΕΝ_έχει_εργαλεία`
- `test_gmail_tool.py::TestCreateEmailDraft::test_δημιουργεί_πρόχειρο_και_ΔΕΝ_στέλνει`
- `test_gmail_tool.py::TestSendToSelf::test_παραλήπτης_από_το_προφίλ`

## Προσθήκη νέου εργαλείου

Όταν προσθέτεις εργαλείο στον agent, χρειάζονται τρία tests:

1. Ότι λειτουργεί σωστά (στο αντίστοιχο `test_*_tool.py`)
2. Ότι είναι καταχωρημένο (`test_graph.py::TestToolRegistration`)
3. Αν γράφει δεδομένα: ότι απαιτεί έγκριση
   (`test_graph.py::TestConfirmationRules`)

Το τρίτο είναι το πιο σημαντικό. Ένα νέο write εργαλείο που ξεχάστηκε
από το `CONFIRMATION_REQUIRED_TOOLS` θα εκτελείται χωρίς να σε ρωτήσει.

## Έλεγχος ότι τα tests πράγματι πιάνουν λάθη

Ένα test suite που περνάει πάντα δεν αποδεικνύει τίποτα. Δοκίμασε να
σπάσεις επίτηδες κάτι και βεβαιώσου ότι κάποιο test κοκκινίζει:

```bash
# Αφαίρεσε προσωρινά το "send_email" από το CONFIRMATION_REQUIRED_TOOLS
# στο app/agents/graph.py και τρέξε:
pytest

# Θα πρέπει να αποτύχουν ~10 tests. Μετά επανάφερέ το.
```
