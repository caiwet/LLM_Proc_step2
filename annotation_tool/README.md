# 🏥 ClinicalAnnotate — Concept Review Tool

A human-friendly annotation tool for reviewing extracted clinical concepts
from NLP pipelines. Supports parquet and CSV datasets.

## Project Structure

```
annotation_tool/
├── app.py                    ← Flask backend
├── index.html                ← Full frontend (single file)
├── data/
│   └── extracted_concepts.csv  ← Your parquet/CSV dataset goes here
│   └── clinical_notes.txt  ← Clinical note texts (optional)
├── annotations/             ← Auto-created, stores JSON annotation files
└── generate_sample.py       ← Sample data generator (for testing)
```

## Setup

### 1. Install dependencies
```bash
pip install flask pandas
# For parquet support:
pip install pyarrow
```

### 2. Add your data
- Drop your `.parquet` or `.csv` file into the `data/` folder.
- Add original text into the `data/` folder.

### 3. Run the server
```bash
cd annotation_tool
python app.py
```

Then open **http://localhost:5001** in your browser.

## Features

### Annotation
- **Per-field voting**: Mark each extracted field as ✓ Correct, ✗ Wrong, or ? Unsure
- **Correction input**: If a field is wrong, enter the correct value
- **Comments**: Add free-text notes per field
- **Row-level shortcuts**: "Mark All Correct" or "Mark Issues" for fast review

### Navigation
- Left sidebar shows all notes with annotation progress badges
- Filter rows: All / Unannotated / Has Issues / Completed
- Keyboard navigation: ← → to move between notes

### Saving
- Annotations auto-save to `annotations/<note_id>_annotations.json`
- Debounced saves (600ms delay) — no manual save button needed

### Export
- Click **↓ Export CSV** in the top bar
- Downloads `annotations_export.csv` with columns:
  `note_id, row_key, field, correct, corrected_value, comment`

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `→` | Next note |
| `←` | Previous note |
| `Ctrl+S` | Confirm save (auto-saves anyway) |
| `Esc` | Close modal |

## Annotation JSON Format

Each `annotations/<note_id>_annotations.json` file looks like:
```json
{
  "row_NOTE001-MED-001": {
    "resourceType": { "correct": true, "corrected_value": "", "comment": "" },
    "status":       { "correct": false, "corrected_value": "active", "comment": "Was marked completed incorrectly" }
  },
  "_meta": {
    "note_id": "NOTE001",
    "last_updated": "2024-10-15T14:32:00Z"
  }
}
```
