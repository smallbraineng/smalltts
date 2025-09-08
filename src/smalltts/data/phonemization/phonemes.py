import logging
import re
from typing import List

from phonemizer.backend import EspeakBackend
from phonemizer.logger import get_logger

from .normalizer import EnglishTextNormalizer

_punct = ';:,.!?¡¿—…"«»"" '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
_syms = []
_seen = set()
for ch in _punct + _letters + _letters_ipa:
    if ch not in _seen:
        _seen.add(ch)
        _syms.append(ch)

p2idx = {ch: i + 1 for i, ch in enumerate(_syms)}
idx2p = {v: k for k, v in p2idx.items()}
phoneme_len = len(p2idx) + 1
phonemes: List[str] = _syms

logging.getLogger().setLevel(logging.CRITICAL)

_es = EspeakBackend(
    language="en-us",
    preserve_punctuation=True,
    with_stress=True,
    words_mismatch="ignore",
    logger=get_logger(verbosity="quiet"),
)
_tok = re.compile(r"\w+|[^\w\s]")
normalizer = EnglishTextNormalizer()


def _phonemize(text: str) -> str:
    text = normalizer.normalize(text)
    phonemized = " ".join(_tok.findall(_es.phonemize([text])[0]))
    return phonemized


def get_token_ids(text: str):
    s = _phonemize(text)
    return [p2idx[c] for c in s if c in p2idx]


def decode_token_ids(token_ids):
    return "".join(idx2p.get(t, "") for t in token_ids)


if __name__ == "__main__":
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world!",
        "Python is an amazing programming language.",
        "Grapheme to phoneme conversion is interesting.",
        "Fantabulousigility is not a real word.",
        "Testing edge cases with special characters: @#$$%^&*()!",
        "Czóloche! Is this a word?",
        "Can you decode this sentence correctly?!",
        "Dr. Smith and Mrs. Johnson met at 3:30pm.",
        "The company earned $1,250,000.50 in Q4 2023.",
        "About 75% of students scored above 90th percentile.",
        "The recipe calls for 1/2 cup sugar and 3/4 tsp salt.",
        "Call me at 555-1234 ext. 42.",
        "The temperature is 98.6°F today.",
        "BTW, the meeting is at 2nd St. near Fort Collins.",
        "£500 equals approximately $625.50.",
        "The 21st century began on January 1st, 2001.",
        "Mr. Rogers lived at 123 Main St., Apt. 4B.",
        "Gen. MacArthur and Lt. Col. Smith discussed plans.",
        "The Rev. Dr. Martin Luther King Jr. gave a speech.",
        "Microsoft Co. Ltd. was founded in 1975.",
        "We need 1,000,000 units by Dec. 31st.",
        "The fraction 7/8 is greater than 3/4.",
    ]
    for s in sentences:
        p = _phonemize(s)
        tids = get_token_ids(s)
        dec = decode_token_ids(tids)
        print("orig   :", s)
        print("phoneme:", p)
        print("tids   :", tids[:64], "..." if len(tids) > 64 else "")
        print("decoded:", dec, "\n")
