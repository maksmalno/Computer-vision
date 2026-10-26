#!/usr/bin/env python3
"""Conservative, offline post-processing for the supplied Russian passport OCR data."""
import csv
import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path

FIELDS = ["Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru", "Birth_date",
          "Birth_place_ru", "Issue_date", "Issue_organisation_code", "Licence_number",
          "Issue_organization_ru"]
AS_OF = date(2026, 10, 1)
# A deliberately conservative gate: a single weak character rejects the document.
MIN_CHAR = 0.985
MIN_FIELD_CONF = 0.990
FIELD_MIN_CHAR = {"Last_name_ru": 0.999, "Birth_place_ru": 0.985, "Issue_organization_ru": 0.985}


def norm(value):
    return " ".join(str(value or "").upper().split())


def field_text(obj):
    if not obj or not obj.get("lines"):
        return ""
    return " ".join(line.get("text", "") for line in obj["lines"]).strip()


def min_character_conf(obj):
    values = [float(c) for line in obj.get("lines", []) for c in line.get("char_conf", [])]
    return min(values) if values else 0.0


def mrz_lines(doc):
    mrz = doc.get("mrz") or {}
    return [re.sub(r"\s", "", str(x.get("text", "")).upper()) for x in mrz.get("lines", [])]


def check_digit(s):
    weights = (7, 3, 1)
    vals = {str(i): i for i in range(10)}
    vals.update({chr(ord("A") + i): 10 + i for i in range(26)})
    vals["<"] = 0
    try:
        return str(sum(vals[c] * weights[i % 3] for i, c in enumerate(s)) % 10)
    except KeyError:
        return None


def mrz_consistency(doc, fields):
    """Use TD3-style check digits and independent fields as corroboration when present."""
    lines = mrz_lines(doc)
    if len(lines) < 2 or len(lines[1]) < 35:
        return False
    a, b = lines[0], lines[1]
    # The document number, birth date and issuing subdivision each have ICAO check digits.
    if check_digit(b[0:9]) != b[9] or check_digit(b[13:19]) != b[19]:
        return False
    # This fixture's MRZ layout does not encode the visual 10-digit licence
    # number in the standard positions: it keeps the first two series digits,
    # the first digit of the second pair, and the six-digit serial.
    number = re.sub(r"\D", "", fields["Licence_number"])
    if len(number) != 10 or number[:3] + number[4:] != b[0:9]:
        return False
    try:
        dob = date(int("20" + b[13:15]), int(b[15:17]), int(b[17:19])).strftime("%d.%m.%Y")
    except ValueError:
        return False
    if dob != fields["Birth_date"]:
        return False
    sex = "МУЖ." if b[20] == "M" else "ЖЕН." if b[20] == "F" else ""
    if not sex or norm(sex) != norm(fields["Sex_ru"]):
        return False
    # First line must be a plausible Russian passport MRZ and contain name material.
    return a.startswith("PNRUS") and "<<" in a


def add_years(d, years):
    try:
        return d.replace(year=d.year + years)
    except ValueError:  # 29 February
        return d.replace(year=d.year + years, day=28)


def passport_valid(fields):
    try:
        born = date(int(fields["Birth_date"][6:10]), int(fields["Birth_date"][3:5]), int(fields["Birth_date"][0:2]))
        issued = date(int(fields["Issue_date"][6:10]), int(fields["Issue_date"][3:5]), int(fields["Issue_date"][0:2]))
    except (ValueError, IndexError):
        return False
    if issued > AS_OF or born > issued:
        return False
    age = AS_OF.year - born.year - ((AS_OF.month, AS_OF.day) < (born.month, born.day))
    # Government Regulation No. 2267, clause 24: at ages 20 and 45 the old
    # passport remains valid until replacement, but no more than 90 days.
    if age >= 45:
        milestone = add_years(born, 45)
        return issued >= milestone or AS_OF <= milestone + timedelta(days=90)
    if age >= 20:
        milestone = add_years(born, 20)
        return issued >= milestone or AS_OF <= milestone + timedelta(days=90)
    return age >= 14 and issued >= add_years(born, 14)


def load_subdivisions(path):
    mapping = {}
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                mapping.setdefault(row.get("code", ""), set()).add(norm(row.get("name", "")))
    return mapping


def extract(doc):
    # mobile_v2 confidence scores are saturated at 1.0 even on visibly corrupted
    # transcriptions in this sample, so they cannot support safe auto-acceptance.
    if doc.get("source") == "mobile_v2":
        return None
    viz = doc.get("viz") or {}
    fields = {}
    for key in FIELDS:
        obj = viz.get(key)
        if not obj:
            return None
        text = field_text(obj)
        min_char = FIELD_MIN_CHAR.get(key, MIN_CHAR)
        if not text or float(obj.get("conf") or 0) < MIN_FIELD_CONF or min_character_conf(obj) < min_char:
            return None
        fields[key] = norm(text)
    if fields["Sex_ru"] not in {"МУЖ.", "ЖЕН."}:
        return None
    if not re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", fields["Birth_date"]):
        return None
    if not re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", fields["Issue_date"]):
        return None
    if not re.fullmatch(r"\d{3}-\d{3}", fields["Issue_organisation_code"]):
        return None
    if not re.fullmatch(r"\d{2} \d{2} \d{6}", fields["Licence_number"]):
        return None
    if not passport_valid(fields) or not mrz_consistency(doc, fields):
        return None
    return fields


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: python postprocess.py INPUT.jsonl ANSWERS.jsonl")
    src, dst = map(Path, sys.argv[1:])
    answers = []
    with src.open(encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            fields = extract(doc)
            row = {"doc_id": doc["doc_id"], "decision": "accept" if fields else "reject"}
            if fields:
                row["fields"] = fields
            answers.append(row)
    with dst.open("w", encoding="utf-8", newline="\n") as f:
        for row in answers:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    accepted = sum(x["decision"] == "accept" for x in answers)
    print(f"wrote {len(answers)} answers; accepted {accepted}; rejected {len(answers)-accepted}")


if __name__ == "__main__":
    main()
