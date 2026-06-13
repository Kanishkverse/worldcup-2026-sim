"""Parse the official FIFA squad-list PDF into squads_2026_real.csv.

Input : SquadLists-English.pdf (48 pages, one national squad per page)
Output: squads_2026_real.csv with one row per player:
        team, shirt_no, position, name, dob, age, club, club_country,
        height_cm, caps, intl_goals

Row shape in the PDF text layer:
    "7 FW MAHREZ Riyad Riyad Karim MAHREZ MAHREZ 21/02/1991 Al Ahli FC (KSA) 179 116 38"
The name block duplicates (player name / first names / last names / shirt
name); we anchor the regex on jersey number + position + DOB + trailing
numeric triple, and keep the leading "SURNAME Given" as the display name.
"""

import csv
import re
import unicodedata
from datetime import date

import pdfplumber

PDF = "SquadLists-English.pdf"
OUT = "squads_2026_real.csv"
KICKOFF = date(2026, 6, 11)

ROW = re.compile(
    r"^(\d{1,2})\s+(GK|DF|MF|FW)\s+(.+?)\s+(\d{2}/\d{2}/\d{4})\s+(.+?)\s+(\d{2,3})\s+(\d{1,3})\s+(\d{1,3})$"
)
TEAM = re.compile(r"^(.+?)\s+\(([A-Z]{3})\)\s*$")

# PDF header names -> the simulator's 2026 field names.
TEAM_ALIASES = {
    "Korea Republic": "South Korea", "Côte d'Ivoire": "Côte d'Ivoire",
    "Ivory Coast": "Côte d'Ivoire", "Czech Republic": "Czechia",
    "Türkiye": "Türkiye", "Turkiye": "Türkiye", "USA": "United States",
    "Cabo Verde": "Cabo Verde", "Cape Verde": "Cabo Verde",
    "Congo DR": "DR Congo", "IR Iran": "Iran", "Korea DPR": "North Korea",
    "Bosnia And Herzegovina": "Bosnia and Herzegovina",
    "Côte D'Ivoire": "Côte d'Ivoire",
}


def display_name(block: str) -> str:
    """'MASTIL Melvin Melvin Feycal MASTIL MASTIL' -> 'Melvin Mastil'."""
    tokens = block.split()
    surname = []
    for tok in tokens:
        if tok.upper() == tok and not tok.istitle():
            surname.append(tok.title())
        else:
            break
    given = tokens[len(surname)] if len(surname) < len(tokens) else ""
    # Mononym players (e.g. "VINICIUS JUNIOR Vinicius ...") repeat the shirt
    # name as the given name, sometimes truncated ("RODRI Rodrigo ..."); drop
    # the duplicate/prefix token (accent-insensitive).
    def _ascii(t: str) -> str:
        return unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()
    g = _ascii(given)
    if given and any(_ascii(s) == g or (len(_ascii(s)) >= 4 and g.startswith(_ascii(s)))
                     for s in surname):
        return " ".join(surname)
    return f"{given} {' '.join(surname)}".strip()


def age_at_kickoff(dob: str) -> int:
    d, m, y = (int(x) for x in dob.split("/"))
    born = date(y, m, d)
    return KICKOFF.year - born.year - ((KICKOFF.month, KICKOFF.day) < (born.month, born.day))


def main() -> None:
    rows = []
    with pdfplumber.open(PDF) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            lines = text.splitlines()
            team = None
            for ln in lines[:8]:
                m = TEAM.match(ln.strip())
                if m:
                    team = TEAM_ALIASES.get(m.group(1).strip(), m.group(1).strip())
                    break
            if not team:
                raise RuntimeError(f"no team header on page {page.page_number}")
            for ln in lines:
                m = ROW.match(ln.strip())
                if not m:
                    continue
                no, pos, names, dob, club_blk, height, caps, goals = m.groups()
                cm = TEAM.match(club_blk)
                club, cc = (cm.group(1), cm.group(2)) if cm else (club_blk, "")
                rows.append({
                    "team": team, "shirt_no": int(no), "position": pos,
                    "name": display_name(names), "dob": dob,
                    "age": age_at_kickoff(dob), "club": club.strip(),
                    "club_country": cc, "height_cm": int(height),
                    "caps": int(caps), "intl_goals": int(goals),
                })

    teams = sorted({r["team"] for r in rows})
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} players across {len(teams)} teams -> {OUT}")
    counts = {t: sum(r['team'] == t for r in rows) for t in teams}
    short = {t: n for t, n in counts.items() if n < 23}
    if short:
        print("teams with <23 parsed players:", short)


if __name__ == "__main__":
    main()
