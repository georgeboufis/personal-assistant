"""
Tests για τα εργαλεία ημερολογίου.

Έμφαση στη ΖΩΝΗ ΩΡΑΣ: εκεί έγιναν τα πραγματικά λάθη κατά την ανάπτυξη
(ο agent είχε βάλει event σε λάθος μέρα, και οι ώρες εμφανίζονταν σε UTC
αντί για ώρα Ελλάδας).
"""

from datetime import datetime
from zoneinfo import ZoneInfo


from app.tools.calendar_tool import (
    create_calendar_event,
    get_events_for_day,
    list_upcoming_events,
    _format_event_start,
)

ATHENS = ZoneInfo("Europe/Athens")


class TestFormatEventStart:
    def test_μετατρέπει_utc_σε_ώρα_ελλάδας(self):
        """15:00 UTC = 18:00 στην Ελλάδα (θερινή ώρα)."""
        result = _format_event_start({"start": {"dateTime": "2026-09-01T15:00:00Z"}})
        assert "18:00" in result

    def test_ώρα_ελλάδας_μένει_ίδια(self):
        result = _format_event_start(
            {"start": {"dateTime": "2026-09-01T18:00:00+03:00"}}
        )
        assert "18:00" in result

    def test_χειμερινή_ώρα_offset_δύο(self):
        """
        Τον Ιανουάριο η Ελλάδα είναι UTC+2, όχι +3. Χρησιμοποιούμε
        ZoneInfo ακριβώς για να το χειρίζεται αυτόματα.
        """
        result = _format_event_start({"start": {"dateTime": "2026-01-15T15:00:00Z"}})
        assert "17:00" in result

    def test_ολοήμερο_event(self):
        result = _format_event_start({"start": {"date": "2026-09-03"}})
        assert "2026-09-03" in result
        assert "ολοήμερο" in result

    def test_περιλαμβάνει_ημέρα_εβδομάδας(self):
        """
        Η ημέρα εβδομάδας βοηθάει τον χρήστη να εντοπίσει λάθος ημερομηνία
        με μια ματιά.
        """
        result = _format_event_start({"start": {"dateTime": "2026-09-01T10:00:00+03:00"}})
        assert "Tuesday" in result


class TestListUpcomingEvents:
    def test_κενό_ημερολόγιο(self, fake_calendar):
        result = list_upcoming_events.invoke({"max_results": 10})
        assert "Δεν υπάρχουν" in result

    def test_εμφανίζει_events(self, fake_calendar):
        fake_calendar.events.return_value.list.return_value.execute.return_value = {
            "items": [
                {"summary": "Ιδιαίτερα AI", "start": {"dateTime": "2026-09-01T15:00:00Z"}},
                {"summary": "Γενέθλια", "start": {"date": "2026-09-03"}},
            ]
        }
        result = list_upcoming_events.invoke({"max_results": 10})

        assert "Ιδιαίτερα AI" in result
        assert "18:00" in result
        assert "Γενέθλια" in result

    def test_event_χωρίς_τίτλο(self, fake_calendar):
        fake_calendar.events.return_value.list.return_value.execute.return_value = {
            "items": [{"start": {"dateTime": "2026-09-01T10:00:00Z"}}]
        }
        assert "(χωρίς τίτλο)" in list_upcoming_events.invoke({"max_results": 5})

    def test_περνάει_το_max_results(self, fake_calendar):
        list_upcoming_events.invoke({"max_results": 3})
        kwargs = fake_calendar.events.return_value.list.call_args.kwargs
        assert kwargs["maxResults"] == 3
        assert kwargs["orderBy"] == "startTime"
        assert kwargs["singleEvents"] is True


class TestCreateCalendarEvent:
    def test_δημιουργεί_με_σωστά_στοιχεία(self, fake_calendar):
        result = create_calendar_event.invoke(
            {
                "summary": "Συνέντευξη Accenture",
                "start_time": "2026-09-09T12:00:00+03:00",
                "end_time": "2026-09-09T13:00:00+03:00",
                "description": "Virtual interview",
            }
        )

        body = fake_calendar.events.return_value.insert.call_args.kwargs["body"]
        assert body["summary"] == "Συνέντευξη Accenture"
        assert body["description"] == "Virtual interview"
        assert body["start"]["dateTime"] == "2026-09-09T12:00:00+03:00"
        assert "Συνέντευξη Accenture" in result

    def test_δηλώνει_ρητά_τη_ζώνη_ώρας(self, fake_calendar):
        """
        Αν το LLM στείλει timestamp χωρίς offset, το ρητό timeZone
        εμποδίζει το Google να το ερμηνεύσει ως UTC.
        """
        create_calendar_event.invoke(
            {
                "summary": "Τεστ",
                "start_time": "2026-09-01T18:00:00",
                "end_time": "2026-09-01T19:00:00",
            }
        )
        body = fake_calendar.events.return_value.insert.call_args.kwargs["body"]
        assert body["start"]["timeZone"] == "Europe/Athens"
        assert body["end"]["timeZone"] == "Europe/Athens"

    def test_επιστρέφει_σύνδεσμο(self, fake_calendar):
        result = create_calendar_event.invoke(
            {"summary": "X", "start_time": "2026-09-01T18:00:00+03:00",
             "end_time": "2026-09-01T19:00:00+03:00"}
        )
        assert "calendar.google.com" in result


class TestGetEventsForDay:
    """
    Χρησιμοποιείται από την πρωινή ενημέρωση. Δεν είναι @tool - δεν το
    βλέπει το LLM.
    """

    def test_ζητά_ακριβώς_τη_ζητούμενη_ημέρα(self, fake_calendar):
        day = datetime(2026, 9, 1, 14, 30, tzinfo=ATHENS)
        get_events_for_day(day)

        kwargs = fake_calendar.events.return_value.list.call_args.kwargs
        assert kwargs["timeMin"].startswith("2026-09-01T00:00:00")
        assert kwargs["timeMax"].startswith("2026-09-02T00:00:00")

    def test_επιστρέφει_λίστα_γραμμών(self, fake_calendar):
        fake_calendar.events.return_value.list.return_value.execute.return_value = {
            "items": [{"summary": "Meeting", "start": {"dateTime": "2026-09-01T07:00:00Z"}}]
        }
        lines = get_events_for_day(datetime(2026, 9, 1, tzinfo=ATHENS))

        assert len(lines) == 1
        assert "Meeting" in lines[0]
        assert "10:00" in lines[0]

    def test_περιλαμβάνει_τοποθεσία(self, fake_calendar):
        fake_calendar.events.return_value.list.return_value.execute.return_value = {
            "items": [
                {
                    "summary": "Συνέντευξη",
                    "start": {"dateTime": "2026-09-09T09:00:00Z"},
                    "location": "Zoom",
                }
            ]
        }
        lines = get_events_for_day(datetime(2026, 9, 9, tzinfo=ATHENS))
        assert "Zoom" in lines[0]

    def test_κενή_μέρα(self, fake_calendar):
        assert get_events_for_day(datetime(2026, 9, 1, tzinfo=ATHENS)) == []
