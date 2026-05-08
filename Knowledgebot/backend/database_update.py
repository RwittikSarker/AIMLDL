import json
import sys
import os
import pandas as pd
from pathlib import Path


OUTPUT_FILE = Path(__file__).parent / "knowledge_base.json"
 
 
def pick_column(columns, prompt):
    """Show numbered list of columns, let user pick one by number."""
    print(f"\n{prompt}")
    for i, col in enumerate(columns):
        print(f"  [{i}] {col}")
    while True:
        try:
            choice = int(input("Enter number: ").strip())
            if 0 <= choice < len(columns):
                return columns[choice]
            print(f"  Please enter a number between 0 and {len(columns) - 1}")
        except ValueError:
            print("  Please enter a valid number.")
 
 
def pick_content_columns(columns, title_col):
    """Let user pick which columns form the content (default = all except title)."""
    remaining = [c for c in columns if c != title_col]
 
    print(f"\nWhich columns should become the Content?")
    print(f"  [A] All remaining columns ({', '.join(remaining)})")
    print(f"  [M] Let me pick manually")
    choice = input("Enter A or M: ").strip().upper()
 
    if choice == "A":
        return remaining
 
    # Manual selection
    print("\nAvailable columns (excluding title):")
    for i, col in enumerate(remaining):
        print(f"  [{i}] {col}")
    print("Enter the numbers separated by commas (e.g. 0,2,3):")
    raw = input("Your selection: ").strip()
    indices = [int(x.strip()) for x in raw.split(",")]
    selected = [remaining[i] for i in indices if 0 <= i < len(remaining)]
    return selected
 
 
def row_to_content(row, content_cols):
    """Combine selected columns into one content string."""
    parts = []
    for col in content_cols:
        value = str(row[col]).strip()
        if value and value.lower() != "nan":
            # Format as "Column Name: value" so context is clear
            parts.append(f"{col}: {value}")
    return "\n".join(parts)
 
 
def main():
    # ── 1. Get CSV file path ──────────────────────────────────────────────────
    if len(sys.argv) < 2:
        csv_path = input("Enter path to your CSV file: ").strip().strip('"')
    else:
        csv_path = sys.argv[1]
 
    if not os.path.exists(csv_path):
        print(f"\n✗ File not found: {csv_path}")
        sys.exit(1)
 
    # ── 2. Load CSV ───────────────────────────────────────────────────────────
    print(f"\nReading {csv_path} ...")
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin-1")  # fallback for some CSVs
 
    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"\nColumns found: {list(df.columns)}")
 
    # Drop completely empty rows
    df = df.dropna(how="all")
    print(f"  (After removing empty rows: {len(df)} rows)")
 
    columns = list(df.columns)
 
    # ── 3. Pick title column ──────────────────────────────────────────────────
    title_col = pick_column(columns, "Which column should be the TITLE of each KB entry?")
    print(f"  → Title column: '{title_col}'")
 
    # ── 4. Pick content columns ───────────────────────────────────────────────
    content_cols = pick_content_columns(columns, title_col)
    print(f"  → Content columns: {content_cols}")
 
    # ── 5. Build KB entries ───────────────────────────────────────────────────
    print("\nBuilding knowledge base entries...")
    entries = []
    skipped = 0
 
    for _, row in df.iterrows():
        title   = str(row[title_col]).strip()
        content = row_to_content(row, content_cols)
 
        # Skip rows where title or content is empty/NaN
        if not title or title.lower() == "nan":
            skipped += 1
            continue
        if not content:
            skipped += 1
            continue
 
        entries.append({"title": title, "content": content})
 
    print(f"✓ Built {len(entries)} entries  ({skipped} rows skipped due to empty data)")
 
    # ── 6. Preview first 3 entries ────────────────────────────────────────────
    print("\n── Preview (first 3 entries) ────────────────────────────")
    for e in entries[:3]:
        print(f"\n  Title:   {e['title'][:80]}")
        print(f"  Content: {e['content'][:120]}...")
    print("─────────────────────────────────────────────────────────")
 
    # ── 7. Confirm before saving ──────────────────────────────────────────────
    confirm = input(f"\nSave {len(entries)} entries to {OUTPUT_FILE}? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled. Nothing was saved.")
        sys.exit(0)
 
    # ── 8. Merge or overwrite ─────────────────────────────────────────────────
    if OUTPUT_FILE.exists():
        merge = input("knowledge_base.json already exists. Merge with existing entries? (y = merge, n = overwrite): ").strip().lower()
        if merge == "y":
            existing = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
            entries  = existing + entries
            print(f"  Merging: {len(existing)} existing + {len(entries) - len(existing)} new = {len(entries)} total")
 
    # ── 9. Write JSON ─────────────────────────────────────────────────────────
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
 
    print(f"\n✓ Done! {len(entries)} entries saved to:")
    print(f"  {OUTPUT_FILE.resolve()}")
    print("\nRestart your backend (python main.py) to load the new knowledge base.")
 
 
if __name__ == "__main__":
    main()
