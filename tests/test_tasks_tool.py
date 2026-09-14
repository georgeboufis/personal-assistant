"""
Tests για τα εργαλεία λίστας εργασιών (Google Tasks).

Ιδιαίτερη προσοχή στην ΠΡΟΘΕΣΜΙΑ: το Google Tasks API δέχεται πλήρες
timestamp αλλά αγνοεί σιωπηλά την ώρα, κρατώντας μόνο την ημερομηνία.
Αν δεν το χειριστούμε ρητά, μια προθεσμία "Παρασκευή 23:00 ώρα Ελλάδας"
μπορεί να καταλήξει Πέμπτη σε UTC.
"""

import pytest

from app.tools.tasks_tool import (
    complete_task,
    create_task,
    delete_task,
    get_pending_tasks,
    list_tasks,
    _format_task,
    _normalize_due_date,
)


class TestNormalizeDueDate:
    def test_απλή_ημερομηνία(self):
        assert _normalize_due_date("2026-09-05") == "2026-09-05T00:00:00.000Z"

    def test_πετάει_την_ώρα(self):
        """
        Η ώρα αγνοείται από το API ούτως ή άλλως. Την κόβουμε εμείς ώστε
        να μη μετατοπιστεί η ημερομηνία λόγω ζώνης ώρας.
        """
        assert _normalize_due_date("2026-09-05T23:00:00+03:00") == "2026-09-05T00:00:00.000Z"

    def test_άκυρη_ημερομηνία_σκάει_καθαρά(self):
        """Καλύτερα σαφές σφάλμα εδώ, παρά κρυπτικό 400 από το Google."""
        with pytest.raises(ValueError):
            _normalize_due_date("αύριο")


class TestFormatTask:
    def test_βασικά_πεδία(self):
        result = _format_task({"id": "t1", "title": "Αποστολή CV"})
        assert "ID: t1" in result
        assert "Αποστολή CV" in result

    def test_δείχνει_μόνο_ημερομηνία_προθεσμίας(self):
        result = _format_task(
            {"id": "t1", "title": "X", "due": "2026-09-05T00:00:00.000Z"}
        )
        assert "2026-09-05" in result
        assert "00:00:00" not in result

    def test_χωρίς_τίτλο(self):
        assert "(χωρίς τίτλο)" in _format_task({"id": "t1"})

    def test_περιλαμβάνει_σημειώσεις(self):
        result = _format_task({"id": "t1", "title": "X", "notes": "από hr@accenture.com"})
        assert "hr@accenture.com" in result


class TestListTasks:
    def test_κενή_λίστα(self, fake_tasks):
        assert "Δεν υπάρχουν εργασίες" in list_tasks.invoke({})

    def test_εμφανίζει_εργασίες(self, fake_tasks):
        fake_tasks.tasks.return_value.list.return_value.execute.return_value = {
            "items": [
                {"id": "t1", "title": "Αποστολή CV", "due": "2026-09-05T00:00:00.000Z"},
                {"id": "t2", "title": "Πληρωμή λογαριασμού"},
            ]
        }
        result = list_tasks.invoke({})

        assert "Αποστολή CV" in result
        assert "2026-09-05" in result
        assert "Πληρωμή λογαριασμού" in result

    def test_κρύβει_ολοκληρωμένες_από_προεπιλογή(self, fake_tasks):
        list_tasks.invoke({})
        kwargs = fake_tasks.tasks.return_value.list.call_args.kwargs
        assert kwargs["showCompleted"] is False

    def test_μπορεί_να_δείξει_ολοκληρωμένες(self, fake_tasks):
        fake_tasks.tasks.return_value.list.return_value.execute.return_value = {
            "items": [{"id": "t1", "title": "Έγινε", "status": "completed"}]
        }
        result = list_tasks.invoke({"include_completed": True})

        kwargs = fake_tasks.tasks.return_value.list.call_args.kwargs
        assert kwargs["showCompleted"] is True
        assert "ολοκληρωμένη" in result

    def test_χρησιμοποιεί_την_προεπιλεγμένη_λίστα(self, fake_tasks):
        list_tasks.invoke({})
        assert fake_tasks.tasks.return_value.list.call_args.kwargs["tasklist"] == "@default"


class TestCreateTask:
    def test_δημιουργεί_με_τίτλο(self, fake_tasks):
        result = create_task.invoke({"title": "Αποστολή CV"})

        body = fake_tasks.tasks.return_value.insert.call_args.kwargs["body"]
        assert body["title"] == "Αποστολή CV"
        assert "Αποστολή CV" in result

    def test_με_προθεσμία(self, fake_tasks):
        create_task.invoke({"title": "X", "due_date": "2026-09-05"})
        body = fake_tasks.tasks.return_value.insert.call_args.kwargs["body"]
        assert body["due"] == "2026-09-05T00:00:00.000Z"

    def test_χωρίς_προθεσμία_δεν_στέλνει_το_πεδίο(self, fake_tasks):
        """Κενό string δεν πρέπει να καταλήξει ως άκυρη προθεσμία."""
        create_task.invoke({"title": "X"})
        body = fake_tasks.tasks.return_value.insert.call_args.kwargs["body"]
        assert "due" not in body

    def test_με_σημειώσεις(self, fake_tasks):
        create_task.invoke({"title": "X", "notes": "από email της Accenture"})
        body = fake_tasks.tasks.return_value.insert.call_args.kwargs["body"]
        assert body["notes"] == "από email της Accenture"


class TestCompleteTask:
    def test_σημειώνει_ολοκληρωμένη(self, fake_tasks):
        complete_task.invoke({"task_id": "t1"})

        kwargs = fake_tasks.tasks.return_value.patch.call_args.kwargs
        assert kwargs["task"] == "t1"
        assert kwargs["body"] == {"status": "completed"}

    def test_χρησιμοποιεί_patch_όχι_update(self, fake_tasks):
        """
        Το patch στέλνει μόνο το πεδίο που αλλάζει. Με update θα έπρεπε να
        ξαναστείλουμε τα πάντα - και ό,τι ξεχνούσαμε θα σβηνόταν.
        """
        complete_task.invoke({"task_id": "t1"})
        fake_tasks.tasks.return_value.patch.assert_called_once()
        fake_tasks.tasks.return_value.update.assert_not_called()

    def test_επιστρέφει_τον_τίτλο(self, fake_tasks):
        result = complete_task.invoke({"task_id": "t1"})
        assert "Δοκιμαστική" in result


class TestDeleteTask:
    def test_διαγράφει(self, fake_tasks):
        delete_task.invoke({"task_id": "t1"})
        kwargs = fake_tasks.tasks.return_value.delete.call_args.kwargs
        assert kwargs["task"] == "t1"
        assert kwargs["tasklist"] == "@default"


class TestGetPendingTasks:
    """Χρησιμοποιείται από την πρωινή ενημέρωση. Δεν είναι @tool."""

    def test_επιστρέφει_λίστα_γραμμών(self, fake_tasks):
        fake_tasks.tasks.return_value.list.return_value.execute.return_value = {
            "items": [
                {"id": "t1", "title": "Αποστολή CV", "due": "2026-09-05T00:00:00.000Z"},
                {"id": "t2", "title": "Χωρίς προθεσμία"},
            ]
        }
        lines = get_pending_tasks()

        assert len(lines) == 2
        assert "Αποστολή CV" in lines[0]
        assert "2026-09-05" in lines[0]
        assert lines[1] == "Χωρίς προθεσμία"

    def test_μόνο_εκκρεμείς(self, fake_tasks):
        get_pending_tasks()
        assert fake_tasks.tasks.return_value.list.call_args.kwargs["showCompleted"] is False

    def test_κενή_λίστα(self, fake_tasks):
        assert get_pending_tasks() == []
