# Gerenated by ChatGPT on 2024-06-19 18:45:00
import numpy as np


LANGUAGE_MAP = {
    # Core Western Languages
    "English": "en", "Englisch": "en", "Anglais": "en",
    "Español": "es", "Spanish": "es", "Castellano": "es",
    "Deutsch": "de", "German": "de",
    "Français": "fr", "French": "fr",
    "Italiano": "it", "Italian": "it",
    "Português": "pt", "Portuguese": "pt",
    "Polski": "pl", "Polish": "pl",
    "Magyar": "hu", "Hungarian": "hu",
    "Română": "ro", "Romanian": "ro",
    "Nederlands": "nl", "Dutch": "nl",
    "ελληνικά": "el", "Greek": "el",
    "русский": "ru", "Pусский": "ru", "Russian": "ru",
    "Український": "uk", "Ukrainian": "uk",

    # East Asian
    "日本語": "ja", "Japanese": "ja",
    "普通话": "zh", "中文": "zh", "Mandarin": "zh",
    "广州话 / 廣州話": "yue", "Cantonese": "yue",
    "한국어/조선말": "ko", "Korean": "ko",

    # South & Middle East
    "हिन्दी": "hi", "Hindi": "hi",
    "தமிழ்": "ta", "Tamil": "ta",
    "తెలుగు": "te", "Telugu": "te",
    "اردو": "ur", "Urdu": "ur",
    "العربية": "ar", "Arabic": "ar",
    "עִבְרִית": "he", "Hebrew": "he",
    "فارسی": "fa", "Farsi": "fa", "Persian": "fa",
    "বাংলা": "bn", "Bengali": "bn",
    "پښتو": "ps", "Pashto": "ps",
    "اردو,": "ur",

    # Nordic & Northern European
    "svenska": "sv", "Swedish": "sv",
    "Dansk": "da", "Danish": "da",
    "Norsk": "no", "Norwegian": "no",
    "Íslenska": "is", "Icelandic": "is",
    "suomi": "fi", "Finnish": "fi",

    # Central / Eastern Europe
    "čeština": "cs", "Český": "cs", "Czech": "cs",
    "Slovenčina": "sk", "Slovak": "sk",
    "Slovenščina": "sl", "Slovenian": "sl",
    "Hrvatski": "hr", "Croatian": "hr",
    "Srpski": "sr", "Serbian": "sr",
    "Bosanski": "bs", "Bosnian": "bs",
    "Eesti": "et", "Estonian": "et",
    "Latviešu": "lv", "Latvian": "lv",
    "Lietuviakai": "lt", "Lietuvių": "lt", "Lithuanian": "lt",
    "български език": "bg", "Bulgarian": "bg",
    "Slovenščina,": "sl",

    # Western & Southern Europe
    "Galego": "gl", "Galician": "gl",
    "Català": "ca", "Catalan": "ca",
    "euskera": "eu", "Basque": "eu",
    "shqip": "sq", "Albanian": "sq",
    "Latina": "la", "Latin": "la",
    "Cymraeg": "cy", "Welsh": "cy",
    "Gaeilge": "ga", "Irish": "ga",

    # African & Creole
    "Afrikaans": "af",
    "isiZulu": "zu",
    "Kiswahili": "sw", "Swahili": "sw",
    "Wolof": "wo",
    "Bamanankan": "bm",
    "Fulfulde": "ff",
    "Somali": "so",
    "Kinyarwanda": "rw",
    "isiXhosa": "xh",

    # Asian & Pacific
    "Tiếng Việt": "vi", "Vietnamese": "vi",
    "Bahasa indonesia": "id", "Bahasa melayu": "ms", "Malay": "ms",
    "ไทย": "th", "ภาษาไทย": "th", "Thai": "th",
    "ქართული": "ka", "Georgian": "ka",
    "Azərbaycan": "az", "Azerbaijani": "az",
    "қазақ": "kk", "Kazakh": "kk",
    "Türkçe": "tr", "Turkish": "tr",
    "Eesti,": "et",
    "Malti": "mt", "Maltese": "mt",

    # Slavic extras
    "Slovenčina,": "sk",
    "беларуская мова": "be", "Belarusian": "be",
    "Srpski,": "sr",

    # Other / Rare / Detected from list
    "Română,": "ro",
    "Pусский,": "ru",
    "Tiếng Việt,": "vi",
    "Latin,": "la",
    "Esperanto": "eo",
    "No Language": "und",
    "??????": "und",
    "?????": "und",
    "Unknown": "und",
    "No Language": "und",
}
