"""
Tests για το εργαλείο καιρού. Δεν κάνουμε ΠΟΤΕ πραγματικές κλήσεις στο
Open-Meteo μέσα στα tests - αντικαθιστούμε το requests.get με ψεύτικες
απαντήσεις, ίδια λογική με τα Google API tests.
"""

from unittest.mock import MagicMock

import pytest

from app.tools.weather_tool import (
    _describe_weather_code,
    _geocode,
    get_weather,
)


def _fake_response(json_data, status_ok=True):
    """Φτιάχνει ένα ψεύτικο requests.Response."""
    resp = MagicMock()
    resp.json.return_value = json_data
    if status_ok:
        resp.raise_for_status.return_value = None
    else:
        resp.raise_for_status.side_effect = Exception("HTTP error")
    return resp


class TestDescribeWeatherCode:
    def test_γνωστός_κωδικός(self):
        assert _describe_weather_code(0) == "αίθριος"
        assert _describe_weather_code(61) == "βροχή"

    def test_άγνωστος_κωδικός_δεν_σκάει(self):
        assert _describe_weather_code(9999) == "άγνωστες συνθήκες"


class TestGeocode:
    def test_επιστρέφει_συντεταγμένες(self, monkeypatch):
        monkeypatch.setattr(
            "app.tools.weather_tool.requests.get",
            lambda *a, **k: _fake_response(
                {"results": [{"latitude": 37.98, "longitude": 23.72, "name": "Αθήνα"}]}
            ),
        )
        lat, lon, name = _geocode("Αθήνα")
        assert lat == 37.98
        assert name == "Αθήνα"

    def test_άγνωστη_τοποθεσία_σκάει_καθαρά(self, monkeypatch):
        monkeypatch.setattr(
            "app.tools.weather_tool.requests.get",
            lambda *a, **k: _fake_response({"results": []}),
        )
        with pytest.raises(ValueError):
            _geocode("ανύπαρκτη πόλη 12345")


class TestGetWeather:
    def test_επιστρέφει_τρέχοντα_καιρό_και_πρόγνωση(self, monkeypatch):
        calls = {"n": 0}

        def fake_get(url, params=None, timeout=None):
            calls["n"] += 1
            if "geocoding" in url:
                return _fake_response(
                    {"results": [{"latitude": 37.98, "longitude": 23.72, "name": "Αθήνα"}]}
                )
            return _fake_response(
                {
                    "current": {
                        "temperature_2m": 28.5,
                        "weather_code": 0,
                        "wind_speed_10m": 12.0,
                    },
                    "daily": {
                        "time": ["2026-09-15", "2026-09-16", "2026-09-17"],
                        "temperature_2m_min": [20.0, 19.5, 21.0],
                        "temperature_2m_max": [30.0, 29.0, 31.0],
                        "weather_code": [0, 2, 61],
                    },
                }
            )

        monkeypatch.setattr("app.tools.weather_tool.requests.get", fake_get)

        result = get_weather.invoke({"location": "Αθήνα"})

        assert "Αθήνα" in result
        assert "28.5°C" in result
        assert "αίθριος" in result
        assert "2026-09-17" in result
        assert "βροχή" in result
        assert calls["n"] == 2  # ένα για geocoding, ένα για forecast

    def test_άγνωστη_τοποθεσία_δεν_σκάει_το_εργαλείο(self, monkeypatch):
        monkeypatch.setattr(
            "app.tools.weather_tool.requests.get",
            lambda *a, **k: _fake_response({"results": []}),
        )
        result = get_weather.invoke({"location": "ανύπαρκτη πόλη"})
        assert "Δεν μπόρεσα να βρω" in result

    def test_προεπιλογή_είναι_αθήνα(self, monkeypatch):
        seen = {}

        def fake_get(url, params=None, timeout=None):
            if "geocoding" in url:
                seen["query"] = params["name"]
                return _fake_response(
                    {"results": [{"latitude": 37.98, "longitude": 23.72, "name": "Athens"}]}
                )
            return _fake_response(
                {
                    "current": {"temperature_2m": 25, "weather_code": 0, "wind_speed_10m": 5},
                    "daily": {
                        "time": ["2026-09-15"],
                        "temperature_2m_min": [20],
                        "temperature_2m_max": [28],
                        "weather_code": [0],
                    },
                }
            )

        monkeypatch.setattr("app.tools.weather_tool.requests.get", fake_get)
        get_weather.invoke({})
        assert seen["query"] == "Athens"