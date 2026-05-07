from pathlib import Path

target = Path("backend/main.py")
text = target.read_text(encoding="utf-8")
em = "\u2014"
en = "\u2013"
em_count = text.count(em)
en_count = text.count(en)
text = text.replace(em, ", ").replace(en, ", ")
target.write_text(text, encoding="utf-8")
print(f"Replaced {em_count} em dashes and {en_count} en dashes in {target}")