
# CAS Story Generator — Streamlit (One-File)

This version is the easiest to deploy. It’s a single `streamlit_app.py` file.

## Run locally
```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501).

## Deploy to Streamlit Community Cloud (free)
1. Push these files to a public GitHub repo.
2. Go to https://share.streamlit.io → **New app** → pick your repo.
3. **Main file path**: `streamlit_app.py`
4. Click **Deploy**. You’ll get a public link you can open on your phone.

## Features
- Targets (phoneme+position+reps), mixed/blocked practice, themes.
- Syllable-shape filters (V, CV, VC, CVC, CCVC, CVCC, CVCV).
- Lexicon CSV upload (or uses bundled sample).
- Preview tables for totals, phrases, per-page coverage.
- One-click PDF download with placeholders & simple icon.
