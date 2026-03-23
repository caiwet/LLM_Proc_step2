#!/usr/bin/env python3
"""
Clinical Annotation Tool - Backend Server
Supports CSV (primary) and Parquet (if pyarrow available)
"""

from flask import Flask, jsonify, request, send_from_directory, send_file
import pandas as pd
import json
import os
import csv
from datetime import datetime

app = Flask(__name__, static_folder='static')

DATA_DIR = 'data'
ANNOTATIONS_DIR = 'annotations'
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def load_dataframe(path):
    """Load CSV or Parquet, return DataFrame."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.parquet':
        try:
            return pd.read_parquet(path)
        except Exception as e:
            raise RuntimeError(f"Cannot read parquet (pyarrow not installed?): {e}")
    elif ext in ('.csv', '.tsv'):
        sep = '\t' if ext == '.tsv' else ','
        return pd.read_csv(path, sep=sep)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def load_annotations(note_id):
    """Load existing annotations for a note."""
    path = os.path.join(ANNOTATIONS_DIR, f"{note_id}_annotations.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def save_annotations(note_id, data):
    """Persist annotations for a note."""
    path = os.path.join(ANNOTATIONS_DIR, f"{note_id}_annotations.json")
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def get_data_files():
    files = []
    if os.path.isdir(DATA_DIR):
        for fn in sorted(os.listdir(DATA_DIR)):
            if fn.endswith(('.csv', '.tsv', '.parquet')):
                stem = os.path.splitext(fn)[0]
                has_note = os.path.exists(os.path.join(DATA_DIR, stem + '.txt'))
                files.append({'name': fn, 'has_note': has_note})
    return files

def get_note_text_for_file(data_filename):
    """
    Given a data filename like 'file1.parquet' or 'file1.csv',
    look for a matching 'file1.txt' in the same data/ directory.
    Returns the note text as a string, or '' if not found.
    """
    stem = os.path.splitext(data_filename)[0]
    txt_path = os.path.join(DATA_DIR, stem + '.txt')
    if os.path.exists(txt_path):
        with open(txt_path, encoding='utf-8') as f:
            return f.read().strip()
    return ''

# ── routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/api/files', methods=['GET'])
def api_files():
    return jsonify({'files': get_data_files()})

@app.route('/api/data/<filename>', methods=['GET'])
def api_data(filename):
    """Return the dataset grouped by note_id."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404
    try:
        df = load_dataframe(path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Replace NaN with empty string for JSON serialisation
    # df = df.where(pd.notna(df), '')
    df = df.where(pd.notna(df), '').astype(str).replace('nan', '')

    note_id = os.path.splitext(filename)[0]
    note_text = get_note_text_for_file(filename)

    return jsonify({
        'columns': df.columns.tolist(),
        'note_ids': [note_id],
        'groups': {
            note_id: {
                'note_text': note_text,
                'rows': df.to_dict(orient='records')
            }
        },
        'total_rows': len(df)
    })

@app.route('/api/annotations/<note_id>', methods=['GET'])
def get_annotations(note_id):
    return jsonify(load_annotations(note_id))

@app.route('/api/annotations/<note_id>', methods=['POST'])
def post_annotations(note_id):
    """
    Payload:
      { row_key: { field: { correct: bool, corrected_value: str, comment: str }, ... }, ... }
    """
    body = request.get_json()
    existing = load_annotations(note_id)
    existing.update(body)
    existing['_meta'] = {
        'last_updated': datetime.utcnow().isoformat() + 'Z',
        'note_id': note_id
    }
    save_annotations(note_id, existing)
    return jsonify({'status': 'saved', 'note_id': note_id})

@app.route('/api/annotations/<note_id>/row', methods=['POST'])
def annotate_row(note_id):
    """Save annotation for a single row."""
    body = request.get_json()
    row_key = body.get('row_key')
    annotation = body.get('annotation')
    if not row_key:
        return jsonify({'error': 'row_key required'}), 400
    existing = load_annotations(note_id)
    existing[row_key] = annotation
    existing['_meta'] = {
        'last_updated': datetime.utcnow().isoformat() + 'Z',
        'note_id': note_id
    }
    save_annotations(note_id, existing)
    return jsonify({'status': 'saved'})

@app.route('/api/export', methods=['GET'])
def export_annotations():
    """Export all annotations as a flat CSV."""
    rows = []
    for fn in os.listdir(ANNOTATIONS_DIR):
        if not fn.endswith('_annotations.json'):
            continue
        with open(os.path.join(ANNOTATIONS_DIR, fn)) as f:
            data = json.load(f)
        note_id = data.get('_meta', {}).get('note_id', fn.replace('_annotations.json', ''))
        for row_key, fields in data.items():
            if row_key == '_meta':
                continue
            if isinstance(fields, dict):
                for field, ann in fields.items():
                    if isinstance(ann, dict):
                        rows.append({
                            'note_id': note_id,
                            'row_key': row_key,
                            'field': field,
                            'correct': ann.get('correct', ''),
                            'corrected_value': ann.get('corrected_value', ''),
                            'comment': ann.get('comment', ''),
                        })
    if not rows:
        return jsonify({'error': 'No annotations yet'}), 404

    import io
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=['note_id','row_key','field','correct','corrected_value','comment'])
    writer.writeheader()
    writer.writerows(rows)
    output.seek(0)
    from flask import Response
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=annotations_export.csv'}
    )

@app.route('/api/progress', methods=['GET'])
def progress():
    """Return annotation progress per note."""
    result = {}
    for fn in os.listdir(ANNOTATIONS_DIR):
        if not fn.endswith('_annotations.json'):
            continue
        with open(os.path.join(ANNOTATIONS_DIR, fn)) as f:
            data = json.load(f)
        note_id = fn.replace('_annotations.json', '')
        ann_rows = {k: v for k, v in data.items() if k != '_meta'}
        result[note_id] = {
            'annotated_rows': len(ann_rows),
            'last_updated': data.get('_meta', {}).get('last_updated', '')
        }
    return jsonify(result)

if __name__ == '__main__':
    print("🏥 Clinical Annotation Tool running at http://localhost:5001")
    app.run(debug=True, port=5001)