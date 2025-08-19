
import streamlit as st
import pandas as pd
import io, re, math, json
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

st.set_page_config(page_title="CAS Story Generator", page_icon="🍪")

# ---------- Data ----------
@st.cache_data
def load_lexicon(df_bytes=None):
    if df_bytes is not None:
        return pd.read_csv(io.BytesIO(df_bytes))
    # fallback to bundled file
    return pd.read_csv("cas_lexicon_expanded.csv")

def get_candidates(df, phoneme, position, shapes):
    pos_col = {"initial":"initial_phonemes","medial":"medial_phonemes","final":"final_phonemes"}[position]
    m = df[pos_col].astype(str).str.startswith(phoneme) & df["syllable_shape"].isin(shapes)
    return df[m]

def build_line_from_targets(df, targets, max_words=6, avoid=None, shapes=("CV","CVC")):
    avoid = set(avoid or [])
    words = []
    pools = {}
    for (ph,pos,count) in targets:
        pool = get_candidates(df, ph, pos, shapes)
        pools[(ph,pos)] = [w for w in pool["word"].tolist() if w not in avoid]
    seq = []
    for ph,pos,count in targets:
        seq += [(ph,pos)]*max(0,int(count))
    for ph,pos in seq:
        pool = pools.get((ph,pos), [])
        pick = None
        for w in pool:
            if w not in words and len((" ".join(words+[w])).split()) <= max_words:
                pick = w; break
        if pick is None and pool:
            pick = pool[0]
        if pick:
            words.append(pick)
        if len(words)>=max_words: break
    return " ".join(words[:max_words]).strip()

def footnote_for_targets(targets):
    cue_bank = {
        "w": "Circle lips for /w/ (woo).",
        "k": "Back of tongue lifts for /k/.",
        "t": "Tip-tap tongue for /t/.",
        "prosody": "Clap the beat: TA-ta."
    }
    cues = []
    seen = set()
    for ph,pos,_ in targets:
        if ph in seen: continue
        seen.add(ph)
        if ph in cue_bank:
            cues.append(f"{ph}: {cue_bank[ph]}")
    cues.append(cue_bank["prosody"])
    return "FOOTNOTE: " + " | ".join(cues[:4])

def make_plan(mode, pages, phrases, targets):
    norm = []
    for t in targets:
        ph = t["phoneme"].strip()
        pos = t["position"].strip().lower()
        reps = int(t.get("reps_per_page",3))
        if ph and pos in {"initial","medial","final"}:
            norm.append((ph,pos,reps))
    if not norm:
        norm = [("w","initial",4),("k","final",3)]
    plan = []
    for i in range(1, pages+1):
        req = norm if mode=="mixed" else [norm[(i-1)%len(norm)]]
        phs = []
        if i in (1,4,7,10): phs.append("I want a cookie")
        elif i in (2,5,8): phs.append("I go")
        elif i in (3,6,9): phs.append("You go")
        if i in (4,8): phs.append("Out")
        plan.append({"page":i, "targets":req, "phrases":phs})
    return plan

def analyze_coverage(story_text, targets, phrases, lex_df):
    pages = {}
    cur = None
    for line in story_text.splitlines():
        m = re.match(r"Page\s+(\d+)", line.strip(), re.I)
        if m:
            cur = int(m.group(1)); pages[cur] = {"lines": [], "footnotes": []}; continue
        if cur is None: continue
        if line.strip().startswith("FOOTNOTE:"):
            pages[cur]["footnotes"].append(line.strip())
        elif line.strip():
            pages[cur]["lines"].append(line.strip())
    per_page, totals = [], {f"{t['phoneme']}_{t['position']}":0 for t in targets}
    for p, data in pages.items():
        text = " ".join(data["lines"]).lower()
        words = re.findall(r"[a-z']+", text)
        counts, unknown = {}, []
        for w in words:
            row = lex_df[lex_df["word"]==w].head(1)
            if row.empty: unknown.append(w); continue
            info = row.iloc[0].to_dict()
            for t in targets:
                ph = t["phoneme"]; pos=t["position"]
                key = f"{ph}_{pos}"
                if str(info[pos+"_phonemes"]).startswith(ph):
                    counts[key] = counts.get(key,0)+1
        for k,v in counts.items():
            totals[k] = totals.get(k,0)+v
        per_page.append({"page":p,"counts":counts,"unknown":unknown})
    all_text = " ".join([" ".join(v["lines"]) for v in pages.values()]).lower()
    phrase_counts = {ph: all_text.count(ph.lower()) for ph in phrases}
    return {"per_page":sorted(per_page,key=lambda x:x["page"]), "totals":totals, "phrase_counts":phrase_counts}

# --- simple icon drawing in PDF ---
def draw_icon(c, label, x, y, w, h, word=None):
    c.rect(x, y, w, h)
    c.setFont("Helvetica-Oblique", 10)
    c.drawCentredString(x+w/2, y+h-12, f"Illustration: {label}")
    # Minimal cookie icon
    if (word or "").lower() in ("cookie","cookies"):
        cx, cy = x+w/2, y+h/2; r=min(w,h)*0.25
        c.circle(cx, cy, r)
        for i in range(6):
            ang = i*60
            c.circle(cx + 0.5*r*math.cos(math.radians(ang)), cy + 0.5*r*math.sin(math.radians(ang)), r*0.08, fill=1)

def story_to_pdf(title, story_text, theme, page_keywords):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 26)
    c.drawCentredString(width/2, height/2 + 20, title)
    c.setFont("Helvetica", 12)
    c.drawCentredString(width/2, height/2 - 10, f"Theme: {theme}")
    c.rect(2.25*inch, height/2 - 2.2*inch, 3.5*inch, 1.5*inch)
    c.setFont("Helvetica-Oblique", 10)
    c.drawCentredString(2.25*inch + 1.75*inch, height/2 - 2.2*inch + 0.75*inch, "Illustration: cover")
    c.showPage()

    prompts = {
        "cookies":["cookie jar","plate of cookies","crumb trail","cookie box","sharing cookies"],
        "park":["swing set","slide","ball game","bench and tree","picnic"],
        "pets":["paw print","kitten","dog bowl","cat nap","pet friends"],
        "space":["rocket ship","moon and stars","astronaut wave","planet ring","counting stars"],
        "farm":["red barn","cow and calf","tractor","hay stack","farm friends"],
    }.get(theme.lower(), ["picture"]*5)

    lines = story_text.splitlines()
    block = []; page_ix = 0
    def draw_block(block_lines, page_ix, word):
        y = height - 1.2*inch
        label = prompts[page_ix % len(prompts)]
        draw_icon(c, label, 1*inch, y - 1.7*inch, width - 2*inch, 1.2*inch, word=word)
        y -= 2.0*inch
        for b in block_lines:
            if b.startswith("FOOTNOTE:"):
                c.setFont("Helvetica-Oblique", 9)
            elif b.lower().startswith("page "):
                c.setFont("Helvetica-Bold", 16)
            else:
                c.setFont("Helvetica", 22)
            c.drawString(1*inch, y, b[:95]); y -= 0.4*inch
            if y < 1*inch:
                c.showPage(); y = height - 1.2*inch
        c.showPage()

    for ln in lines:
        if ln.strip().lower().startswith("page "):
            if block:
                word = page_keywords[page_ix] if page_ix < len(page_keywords) else None
                draw_block(block, page_ix, word); block=[]; page_ix += 1
            block.append(ln.strip())
        else:
            if ln.strip(): block.append(ln.strip())
    if block:
        word = page_keywords[page_ix] if page_ix < len(page_keywords) else None
        draw_block(block, page_ix, word)

    c.save(); buf.seek(0)
    return buf.getvalue()

# ---------- UI ----------
st.title("🍪 CAS Story Generator (Streamlit)")

with st.sidebar:
    st.header("Settings")
    title = st.text_input("Story title", "Cookie Game — Custom Story")
    pages = st.slider("Pages", 6, 14, 10)
    mode = st.selectbox("Practice mode", ["mixed","blocked"])
    theme = st.selectbox("Theme", ["cookies","park","pets","space","farm"])
    phrases_raw = st.text_input("Core phrases (comma-separated)", "I want a cookie,I go,You go,Out")
    phrases = [p.strip() for p in phrases_raw.split(",") if p.strip()]

    st.markdown("**Targets** (phoneme, position, reps/page)")
    t1 = {"phoneme": st.text_input("t1 phoneme","w"), "position": st.selectbox("t1 position", ["initial","medial","final"], index=0), "reps_per_page": st.number_input("t1 reps/page", 0, 10, 4)}
    t2 = {"phoneme": st.text_input("t2 phoneme","k"), "position": st.selectbox("t2 position", ["initial","medial","final"], index=2), "reps_per_page": st.number_input("t2 reps/page", 0, 10, 3)}
    t3 = {"phoneme": st.text_input("t3 phoneme",""), "position": st.selectbox("t3 position", ["initial","medial","final"], index=0), "reps_per_page": st.number_input("t3 reps/page", 0, 10, 0)}
    targets = [t for t in [t1,t2,t3] if t["phoneme"]]

    st.markdown("**Syllable shapes**")
    shapes = []
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.checkbox("V"): shapes.append("V")
        if st.checkbox("CV", value=True): shapes.append("CV")
    with col2:
        if st.checkbox("VC"): shapes.append("VC")
        if st.checkbox("CVC", value=True): shapes.append("CVC")
    with col3:
        if st.checkbox("CCVC"): shapes.append("CCVC")
        if st.checkbox("CVCC"): shapes.append("CVCC")
        if st.checkbox("CVCV"): shapes.append("CVCV")

    st.markdown("**Lexicon**")
    up = st.file_uploader("Upload lexicon CSV (optional)", type=["csv"])
    lex_df = load_lexicon(up.read() if up else None)

st.subheader("Preview & Validate")

plan = make_plan(mode, pages, phrases, targets)
# generate story
used = set()
page_keywords = []
lines_out = []
for i, spec in enumerate(plan, start=1):
    if phrases:
        lines_out.append(f"Page {i}")
        lines_out.append(spec["phrases"][0] if spec["phrases"] else "")
    line2 = build_line_from_targets(lex_df, spec["targets"], max_words=6, avoid=used, shapes=tuple(shapes or ["CV","CVC"]))
    used.update(line2.split())
    page_keywords.append(line2.split()[0] if line2 else "cookie")
    lines_out.append(line2)
    lines_out.append(footnote_for_targets(spec["targets"]))
    lines_out.append("")

story_text = "\n".join(lines_out)
cov = analyze_coverage(story_text, targets, phrases, lex_df)

with st.expander("Show story text"):
    st.code(story_text, language="markdown")

# Coverage tables
colA, colB = st.columns(2)
with colA:
    st.markdown("**Totals by target**")
    rows = []
    for t in targets:
        key = f"{t['phoneme']}_{t['position']}"
        count = cov["totals"].get(key, 0)
        goal = t["reps_per_page"] * pages
        rows.append({"target": key, "count": count, "goal": goal})
    st.dataframe(pd.DataFrame(rows))

with colB:
    st.markdown("**Phrase counts**")
    st.dataframe(pd.DataFrame([{"phrase": ph, "count": cov["phrase_counts"].get(ph,0)} for ph in phrases]))

st.markdown("**Per-page coverage**")
pp_rows = []
keys = [f"{t['phoneme']}_{t['position']}" for t in targets]
for row in cov["per_page"]:
    r = {"page": row["page"], **{k: row["counts"].get(k,0) for k in keys}, "unknown": ", ".join(row["unknown"][:6])}
    pp_rows.append(r)
st.dataframe(pd.DataFrame(pp_rows))

# Download PDF
pdf_bytes = story_to_pdf(title, story_text, theme, page_keywords)
st.download_button("Download PDF", data=pdf_bytes, file_name="cas_story.pdf", mime="application/pdf")

st.caption("Counts are approximate; rely on clinician judgment. Icons are placeholders.")

