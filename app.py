import json
import math
import os
import re
import sys

from flask import Flask, render_template_string, request
from PyPDF2 import PdfReader

# Only import Streamlit when running the Streamlit UI.
USE_FLASK = any(arg.lower() == "flask" for arg in sys.argv[1:])
if not USE_FLASK:
    import streamlit as st
    import streamlit.components.v1 as components

from openai import OpenAI


def build_client() -> OpenAI:
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set GROQ_API_KEY (recommended) or OPENAI_API_KEY."
        )

    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )


client = build_client()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "replace-with-a-secret-key")

FLASK_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Career Analyzer</title>
    <style>
        body { background: #020617; color: #e2e8f0; font-family: Arial, sans-serif; margin: 0; padding: 0; }
        .container { max-width: 900px; margin: 2rem auto; padding: 1.5rem; background: #0f172a; border-radius: 18px; box-shadow: 0 20px 40px rgba(0,0,0,0.35); }
        h1 { color: #a78bfa; margin-bottom: 1rem; }
        label { display: block; margin-top: 1rem; margin-bottom: 0.5rem; font-size: 0.95rem; color: #94a3b8; }
        input[type=text], input[type=file], textarea { width: 100%; padding: 0.85rem 1rem; border: 1px solid #334155; border-radius: 12px; background: #020617; color: #e2e8f0; }
        button { margin-top: 1.25rem; background: linear-gradient(135deg, #7c3aed, #4f46e5); color: white; border: none; padding: 0.95rem 1.3rem; border-radius: 12px; cursor: pointer; font-size: 0.95rem; }
        .result { margin-top: 1.5rem; padding: 1.25rem; background: #111827; border: 1px solid #334155; border-radius: 14px; }
        pre { white-space: pre-wrap; word-break: break-word; font-family: monospace; color: #dcdcdc; }
        .warning { color: #fbbf24; }
    </style>
</head>
<body>
    <div class="container">
        <h1>AI Career Analyzer</h1>
        <p>Upload a PDF resume, enter the target role and your interests, then submit to score the resume and generate a roadmap.</p>
        <form method="post" enctype="multipart/form-data">
            <label for="resume">Resume (PDF)</label>
            <input type="file" name="resume" id="resume" accept="application/pdf" required>
            <label for="role">Target Job Role</label>
            <input type="text" name="role" id="role" placeholder="e.g. Backend Engineer" required>
            <label for="interest">Your Interests</label>
            <input type="text" name="interest" id="interest" placeholder="e.g. AI, cloud, APIs">
            <button type="submit">Analyze Resume</button>
        </form>
        {% if error %}
            <div class="result warning"><strong>Error:</strong> {{ error }}</div>
        {% endif %}
        {% if step1 %}
            <div class="result">
                <h2>Step 1 - Scoring & Analysis</h2>
                <pre>{{ step1 }}</pre>
            </div>
        {% endif %}
        {% if step2 %}
            <div class="result">
                <h2>Step 2 - Roadmap</h2>
                <pre>{{ step2 }}</pre>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "have", "been",
    "will", "are", "was", "were", "has", "had", "but", "not", "you", "your",
    "our", "their", "they", "them", "then", "than", "also", "into", "over",
    "after", "before", "about", "while", "when", "where", "which", "who",
    "what", "how", "its", "can", "may", "should", "could", "would", "all",
    "any", "each", "both", "more", "most", "other", "such", "well", "just",
    "like", "some", "very", "only", "even", "new", "good", "work", "use",
    "used", "using", "make", "made", "provide", "provided", "ensure",
    "responsible", "including", "across", "multiple", "various", "strong",
    "within", "ability", "excellent", "experience", "years", "year",
    "worked", "developed", "managed", "team", "role", "skills", "skill",
    "knowledge", "time", "based", "support", "during", "help", "key",
    "part", "take", "led", "one", "two", "three", "four", "five", "six",
}

SCORE_DIMENSIONS = [
    "Skills Match",
    "Experience Depth",
    "Education Relevance",
    "Projects & Portfolio",
    "Certifications",
    "Resume Presentation",
]


def clean_resume_text(raw_text: str, max_tokens: int = 800) -> str:
    text = re.sub(r"\s+", " ", raw_text)
    text = re.sub(r"\S+@\S+", "[EMAIL]", text)
    text = re.sub(r"https?://\S+", "[URL]", text)
    text = re.sub(r"\b\d{10}\b", "[PHONE]", text)

    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+#.\-]*|[\d]+%?", text)

    seen = set()
    kept = []
    for token in tokens:
        lower = token.lower()
        if lower not in STOPWORDS and len(token) > 2 and lower not in seen:
            seen.add(lower)
            kept.append(token)

    approx_limit = int(max_tokens * 1.3)
    return " ".join(kept[:approx_limit])


def extract_text(file) -> str:
    if hasattr(file, "seek"):
        file.seek(0)

    reader = PdfReader(file)
    text_parts = []
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text_parts.append(extracted)

    if hasattr(file, "seek"):
        file.seek(0)

    return "\n".join(text_parts)


def render_flask_page(error=None, step1=None, step2=None):
    return render_template_string(FLASK_TEMPLATE, error=error, step1=step1, step2=step2)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploaded = request.files.get("resume")
        role = request.form.get("role", "").strip()
        interest = request.form.get("interest", "").strip()

        if not uploaded or uploaded.filename == "":
            return render_flask_page(error="Please upload a PDF resume.")
        if not role:
            return render_flask_page(error="Please enter a target job role.")

        try:
            raw_text = extract_text(uploaded)
            cleaned_text = clean_resume_text(raw_text)
            step1 = analyze_step1(cleaned_text, role, interest)
            step2 = analyze_step2(cleaned_text, role, interest, step1)
            return render_flask_page(step1=json.dumps(step1, indent=2), step2=step2)
        except Exception as exc:
            return render_flask_page(error=str(exc))

    return render_flask_page()


def _response_text(response) -> str:
    if hasattr(response, "output_text") and response.output_text is not None:
        return response.output_text

    output = getattr(response, "output", None)
    if isinstance(output, list):
        parts = []
        for item in output:
            content = item.get("content") if isinstance(item, dict) else None
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for chunk in content:
                    if isinstance(chunk, dict) and chunk.get("type") == "output_text":
                        parts.append(chunk.get("text", ""))
        if parts:
            return "".join(parts)

    return str(response)


def analyze_step1(cleaned_text: str, role: str, interest: str) -> dict:
    prompt = f"""
You are a senior technical recruiter and career coach.

Target Role: {role}
Candidate Interests: {interest}

Compressed Resume Keywords:
{cleaned_text}

## Your Task
Score this candidate on EXACTLY these 6 dimensions for the role "{role}".
Each score is 0-100 (integer). Be honest and calibrated - not everyone scores 80+.

Dimensions:
1. Skills Match - How well do their listed skills match the role requirements?
2. Experience Depth - Quality and depth of experience (projects, jobs, internships)
3. Education Relevance - Degree/coursework relevance to the target role
4. Projects & Portfolio - Concrete projects demonstrated
5. Certifications - Relevant certs, courses, credentials
6. Resume Presentation - Clarity, structure, quantification of achievements

Also produce:
- overall_score: weighted average (Skills 30%, Experience 25%, Education 15%, Projects 15%, Certs 10%, Presentation 5%)
- readiness_label: one of "Not Ready", "Early Stage", "Developing", "Job-Ready", "Strong Candidate"
- score_justification: 2-sentence overall summary

Return ONLY valid JSON in this exact format (no markdown, no backticks):
{{
  "scores": {{
    "Skills Match": <int>,
    "Experience Depth": <int>,
    "Education Relevance": <int>,
    "Projects & Portfolio": <int>,
    "Certifications": <int>,
    "Resume Presentation": <int>
  }},
  "overall_score": <int>,
  "readiness_label": "<string>",
  "score_justification": "<string>"
}}
"""
    response = client.responses.create(
        model="openai/gpt-oss-20b",
        input=prompt,
    )
    raw = _response_text(response).strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Step 1 returned invalid JSON: {raw}") from exc


def analyze_step2(cleaned_text: str, role: str, interest: str, scores: dict) -> str:
    weak_dims = [key for key, value in scores["scores"].items() if value < 60]
    prompt = f"""
You are a career coach helping a candidate targeting: {role}
Interests: {interest}

Their weak scoring areas are: {", ".join(weak_dims) if weak_dims else "None - they are strong overall"}

Compressed Resume Keywords:
{cleaned_text}

Provide a structured career roadmap in Markdown with EXACTLY these sections:

## Gap Analysis
List 3-5 specific skills or knowledge gaps for the "{role}" role that are NOT evident in the resume.

## Certifications & Courses to Pursue
Recommend 3 specific, real certifications or online courses (with platform names like Coursera, Google, AWS, etc.) that would close the gaps.

## Project Recommendations
Suggest 3 concrete project ideas the candidate can build to strengthen their portfolio for "{role}". Be specific - name technologies, describe the project briefly.

## Internship vs Full-Time Job?
Based on their profile, give a clear recommendation: should they target internships first or apply for full-time roles? Give 2-3 sentences of reasoning.

## Job Matching (Real Roles to Target)
List 5 real job titles (not companies - titles) that match this candidate's current profile AND their target of "{role}". For each, give one sentence on why it's a good stepping stone.

Keep tone direct, practical, and encouraging. No fluff.
"""
    response = client.responses.create(
        model="openai/gpt-oss-20b",
        input=prompt,
    )
    return _response_text(response)


def radar_chart_svg(scores: dict, size: int = 340) -> str:
    labels = SCORE_DIMENSIONS
    values = [scores.get(label, 0) / 100.0 for label in labels]
    count = len(labels)
    center_x = size // 2
    center_y = size // 2
    radius = size // 2 - 60

    def point(angle_deg, ring_radius):
        angle_rad = math.radians(angle_deg - 90)
        x_pos = center_x + ring_radius * math.cos(angle_rad)
        y_pos = center_y + ring_radius * math.sin(angle_rad)
        return x_pos, y_pos

    angles = [index * 360 / count for index in range(count)]

    rings_svg = ""
    for level in [0.2, 0.4, 0.6, 0.8, 1.0]:
        pts = [point(angle, radius * level) for angle in angles]
        polygon = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        opacity = 0.15 if level < 1.0 else 0.3
        rings_svg += (
            f'<polygon points="{polygon}" fill="none" stroke="#a78bfa" '
            f'stroke-width="1" stroke-opacity="{opacity}"/>'
        )

    axes_svg = ""
    for angle in angles:
        outer_x, outer_y = point(angle, radius)
        axes_svg += (
            f'<line x1="{center_x}" y1="{center_y}" x2="{outer_x:.1f}" y2="{outer_y:.1f}" '
            f'stroke="#a78bfa" stroke-opacity="0.2" stroke-width="1"/>'
        )

    data_points = [point(angles[index], radius * values[index]) for index in range(count)]
    data_polygon = " ".join(f"{x:.1f},{y:.1f}" for x, y in data_points)

    labels_svg = ""
    for index, (label, value) in enumerate(zip(labels, values)):
        label_x, label_y = point(angles[index], radius + 30)
        score = int(value * 100)

        if score >= 75:
            color = "#4ade80"
        elif score >= 50:
            color = "#facc15"
        else:
            color = "#f87171"

        anchor = "middle"
        if label_x < center_x - 10:
            anchor = "end"
        elif label_x > center_x + 10:
            anchor = "start"

        labels_svg += f"""
        <text x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="{anchor}" font-size="10" fill="#e2e8f0" font-family="monospace">{label}</text>
        <text x="{label_x:.1f}" y="{label_y + 13:.1f}" text-anchor="{anchor}" font-size="11" font-weight="bold" fill="{color}" font-family="monospace">{score}</text>
        """

    dots_svg = ""
    for dot_x, dot_y in data_points:
        dots_svg += f'<circle cx="{dot_x:.1f}" cy="{dot_y:.1f}" r="4" fill="#a78bfa" stroke="#fff" stroke-width="1.5"/>'

    return f"""
    <!DOCTYPE html>
    <html>
    <body style="margin:0;background:#020617;">
        <svg viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">
            <rect width="{size}" height="{size}" rx="16" fill="#0f172a"/>
            {rings_svg}
            {axes_svg}
            <polygon points="{data_polygon}" fill="#7c3aed" fill-opacity="0.35" stroke="#a78bfa" stroke-width="2"/>
            {dots_svg}
            {labels_svg}
        </svg>
    </body>
    </html>
    """


def score_bar_html(overall: int, label: str) -> str:
    color_map = {
        "Not Ready": "#ef4444",
        "Early Stage": "#f97316",
        "Developing": "#facc15",
        "Job-Ready": "#4ade80",
        "Strong Candidate": "#22c55e",
    }
    bar_color = color_map.get(label, "#a78bfa")

    return f"""
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:20px 24px;margin-bottom:16px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
            <span style="color:#94a3b8;font-size:13px;font-family:monospace;letter-spacing:0.05em;">OVERALL READINESS</span>
            <span style="color:{bar_color};font-size:28px;font-weight:800;font-family:monospace;">{overall}<span style="font-size:16px;color:#64748b;">/100</span></span>
        </div>
        <div style="background:#1e293b;border-radius:99px;height:10px;overflow:hidden;">
            <div style="background:linear-gradient(90deg, {bar_color}88, {bar_color});width:{overall}%;height:100%;border-radius:99px;"></div>
        </div>
        <div style="margin-top:10px;text-align:right;">
            <span style="background:{bar_color}22;color:{bar_color};border:1px solid {bar_color}44;padding:3px 12px;border-radius:99px;font-size:12px;font-family:monospace;font-weight:600;">{label}</span>
        </div>
    </div>
    """


def dimension_bars_html(scores: dict) -> str:
    weights = {
        "Skills Match": 30,
        "Experience Depth": 25,
        "Education Relevance": 15,
        "Projects & Portfolio": 15,
        "Certifications": 10,
        "Resume Presentation": 5,
    }
    parts = ['<div style="display:flex;flex-direction:column;gap:8px;">']
    for dimension in SCORE_DIMENSIONS:
        value = scores.get(dimension, 0)
        weight = weights.get(dimension, 10)
        if value >= 75:
            color = "#4ade80"
        elif value >= 50:
            color = "#facc15"
        else:
            color = "#f87171"

        parts.append(
            f"""
            <div>
                <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span style="font-size:12px;color:#94a3b8;font-family:monospace;">{dimension} <span style="color:#475569;font-size:10px;">(weight {weight}%)</span></span>
                    <span style="font-size:12px;color:{color};font-weight:700;font-family:monospace;">{value}</span>
                </div>
                <div style="background:#1e293b;border-radius:99px;height:6px;overflow:hidden;">
                    <div style="background:{color};width:{value}%;height:100%;border-radius:99px;"></div>
                </div>
            </div>
            """
        )
    parts.append("</div>")
    return "".join(parts)


def render_html_block(html: str, height: int) -> None:
    components.html(
        f"""
        <!DOCTYPE html>
        <html>
        <body style="margin:0;background:#020617;color:#e2e8f0;font-family:Arial,sans-serif;">
            {html}
        </body>
        </html>
        """,
        height=height,
        scrolling=False,
    )


def run_streamlit_ui():
    st.set_page_config(
        page_title="AI Career Analyzer",
        page_icon="rocket",
        layout="wide",
    )

    st.markdown(
        """
<style>
    html, body, [class*="css"] {
        background-color: #020617 !important;
        color: #e2e8f0 !important;
    }
    .stApp { background-color: #020617; }
    .block-container { max-width: 1100px; padding-top: 2rem; }
    h1, h2, h3 { color: #f1f5f9 !important; font-family: monospace; }
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-family: monospace;
        font-size: 14px;
        font-weight: 600;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
    .stTextInput > div > div > input {
        background: #0f172a !important;
        border: 1px solid #1e293b !important;
        color: #e2e8f0 !important;
        border-radius: 8px;
        font-family: monospace;
    }
    .stFileUploader {
        background: #0f172a !important;
        border: 1px dashed #334155 !important;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: monospace;
        color: #64748b;
    }
    .stTabs [aria-selected="true"] {
        color: #a78bfa !important;
        border-bottom-color: #a78bfa !important;
    }
    .stMarkdown { color: #cbd5e1; }
    div[data-testid="stExpander"] {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
    }
</style>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div style="text-align:center; padding: 2rem 0 1rem 0;">
    <div style="font-size:48px; margin-bottom:8px;">rocket</div>
    <h1 style="font-size:2rem; letter-spacing:0.1em; margin:0;">AI CAREER ANALYZER</h1>
    <p style="color:#64748b; font-family:monospace; font-size:13px; margin-top:6px;">
        Upload resume -> Get scored -> Get a roadmap
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    col_a, col_b, col_c = st.columns([2, 1.5, 1.5])
    with col_a:
        resume = st.file_uploader("Upload Resume (PDF)", type="pdf")
    with col_b:
        role = st.text_input("Target Job Role", placeholder="e.g. Backend Engineer")
    with col_c:
        interest = st.text_input("Your Interests", placeholder="e.g. AI, cloud, APIs")

    analyze = st.button("Analyze Resume")

    if resume is not None:
        with st.expander("Token Preview (what AI actually sees)", expanded=False):
            raw = extract_text(resume)
            cleaned = clean_resume_text(raw)
            preview_col_1, preview_col_2 = st.columns(2)
            with preview_col_1:
                st.caption("Raw text (first 300 chars)")
                st.code(raw[:300], language=None)
            with preview_col_2:
                st.caption(f"Cleaned tokens (~{len(cleaned.split())} tokens sent to AI)")
                st.code(cleaned[:400], language=None)

    if analyze:
        if not resume:
            st.warning("Please upload a PDF resume first.")
            return
        if not role:
            st.warning("Please enter a target job role.")
            return

        with st.spinner("Extracting and cleaning resume..."):
            raw_text = extract_text(resume)
            if not raw_text.strip():
                st.error("No readable text was found in the PDF. Try a text-based PDF instead of a scanned image.")
                return
            cleaned_text = clean_resume_text(raw_text)

        tab1, tab2 = st.tabs(["Step 1 - Scoring & Analysis", "Step 2 - Roadmap"])

        with tab1:
            with st.spinner("Scoring your resume with AI..."):
                try:
                    step1 = analyze_step1(cleaned_text, role, interest)
                except Exception as exc:
                    st.error(f"AI scoring failed: {exc}")
                    return

            overall = step1.get("overall_score", 0)
            label = step1.get("readiness_label", "-")
            scores = step1.get("scores", {})
            justification = step1.get("score_justification", "")

            render_html_block(score_bar_html(overall, label), height=135)

            if justification:
                st.markdown(
                    f"""
                <div style="background:#0f172a; border-left:3px solid #a78bfa; padding:12px 16px;
                     border-radius:8px; margin-bottom:16px;">
                    <span style="color:#94a3b8; font-family:monospace; font-size:13px;">{justification}</span>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            col_radar, col_bars = st.columns([1, 1])
            with col_radar:
                st.markdown("**Radar View**")
                components.html(radar_chart_svg(scores), height=360, scrolling=False)

            with col_bars:
                st.markdown("**Dimension Breakdown**")
                render_html_block(dimension_bars_html(scores), height=275)

                with st.expander("How scoring works"):
                    st.markdown(
                        """
**Scoring Model**

| Dimension | Weight | What it measures |
|---|---|---|
| Skills Match | 30% | Tech stack / tools alignment |
| Experience Depth | 25% | Jobs, internships, real usage |
| Education | 15% | Degree relevance |
| Projects | 15% | Portfolio evidence |
| Certifications | 10% | Credentials & courses |
| Presentation | 5% | Clarity, quantified achievements |

Scores are AI-calibrated against typical candidates for the target role.
"""
                    )

        with tab2:
            with st.spinner("Generating your career roadmap..."):
                try:
                    step2_md = analyze_step2(cleaned_text, role, interest, step1)
                except Exception as exc:
                    st.error(f"Roadmap generation failed: {exc}")
                    return

            st.markdown(step2_md)


def run_flask_app():
    ports = [8501, 8502, 5000]
    for port in ports:
        try:
            print(f"Starting Flask on http://127.0.0.1:{port}")
            app.run(host="127.0.0.1", port=port, debug=True)
            return
        except OSError as exc:
            print(f"Port {port} unavailable: {exc}")
            continue
    print("Could not start Flask on any available port.")


if not USE_FLASK:
    run_streamlit_ui()


if __name__ == "__main__":
    if USE_FLASK:
        run_flask_app()
    else:
        print("To run this app:")
        print("  streamlit run app.py")
        print("or to start the Flask upload UI:")
        print("  python app.py flask")
