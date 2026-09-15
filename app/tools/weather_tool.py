"""
weather_tool.py
-----------------
Το εργαλείο καιρού. Χρησιμοποιεί το Open-Meteo API, που είναι εντελώς
δωρεάν και ΔΕΝ χρειάζεται API key - καμία εγγραφή, μηδενικό κόστος,
μηδενικό ρίσκο (μόνο ανάγνωση δημόσιων δεδομένων καιρού, καμία ενέργεια).

Γι' αυτό δεν μπαίνει καθόλου στο CONFIRMATION_REQUIRED_TOOLS - δεν έχει
νόημα να ζητάμε έγκριση για κάτι που απλά διαβάζει μια δημόσια πρόγνωση.
"""

import requests
from langchain_core.tools import tool

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Μετάφραση των WMO weather codes (διεθνές πρότυπο) σε κατανοητή
# περιγραφή. Πλήρης λίστα: https://open-meteo.com/en/docs
WEATHER_DESCRIPTIONS = {
    0: "αίθριος", 1: "κυρίως αίθριος", 2: "μερική συννεφιά", 3: "συννεφιά",
    45: "ομίχλη", 48: "παγωμένη ομίχλη",
    51: "ψιλόβροχο", 53: "ψιλόβροχο", 55: "ψιλόβροχο",
    61: "βροχή", 63: "βροχή", 65: "δυνατή βροχή",
    71: "χιόνι", 73: "χιόνι", 75: "δυνατό χιόνι",
    80: "μπόρες", 81: "μπόρες", 82: "ισχυρές μπόρες",
    95: "καταιγίδα", 96: "καταιγίδα με χαλάζι", 99: "ισχυρή καταιγίδα με χαλάζι",
}


def _describe_weather_code(code: int) -> str:
    return WEATHER_DESCRIPTIONS.get(code, "άγνωστες συνθήκες")


def _geocode(location: str) -> tuple[float, float, str]:
    """
    Μετατρέπει ένα όνομα τοποθεσίας (π.χ. "Αθήνα") σε συντεταγμένες.
    Επιστρέφει (latitude, longitude, κανονικό όνομα όπως το αναγνώρισε η υπηρεσία).
    """
    response = requests.get(
        GEOCODING_URL,
        params={"name": location, "count": 1, "language": "el", "format": "json"},
        timeout=10,
    )
    response.raise_for_status()
    results = response.json().get("results")

    if not results:
        raise ValueError(f"Δεν βρέθηκε τοποθεσία με το όνομα '{location}'.")

    first = results[0]
    return first["latitude"], first["longitude"], first.get("name", location)


@tool
def get_weather(location: str = "Athens") -> str:
    """
    Επιστρέφει τον τρέχοντα καιρό και πρόγνωση 3 ημερών για μια τοποθεσία.

    Χρησιμοποίησε αυτό όταν ο χρήστης ρωτάει για τον καιρό, αν χρειάζεται
    ομπρέλα, τι θερμοκρασία θα κάνει, κλπ. Δωρεάν, δημόσια δεδομένα -
    δεν χρειάζεται έγκριση.

    Args:
        location: το όνομα της πόλης (π.χ. "Αθήνα", "Θεσσαλονίκη",
            "Λονδίνο"). Προεπιλογή: Αθήνα.
    """
    try:
        lat, lon, name = _geocode(location)
    except Exception as exc:
        return f"Δεν μπόρεσα να βρω την τοποθεσία '{location}': {exc}"

    response = requests.get(
        FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,weather_code",
            "timezone": "auto",
            "forecast_days": 3,
        },
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    current = data["current"]
    lines = [
        f"Καιρός στο/στη {name} τώρα:",
        f"- {current['temperature_2m']}°C, "
        f"{_describe_weather_code(current['weather_code'])}, "
        f"άνεμος {current['wind_speed_10m']} km/h",
        "",
        "Πρόγνωση:",
    ]

    daily = data["daily"]
    for i, date in enumerate(daily["time"]):
        lines.append(
            f"- {date}: {daily['temperature_2m_min'][i]:.0f}°C έως "
            f"{daily['temperature_2m_max'][i]:.0f}°C, "
            f"{_describe_weather_code(daily['weather_code'][i])}"
        )

    return "\n".join(lines)