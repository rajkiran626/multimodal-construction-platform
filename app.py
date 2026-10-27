"""
Multimodal Construction Intelligence Platform
Skill Wallet Capstone Project 1

Modules
  1. Site Safety & PPE Audit            (Epic 2)
  2. Crack Diagnostics & Integrity      (Epic 2)
  3. Planning & Real Estate Valuation   (Epic 3)
  4. Daily Site Assistant + Weather     (Epic 5)

Stack: Streamlit + Google Gemini (google-genai SDK) + Pillow + Pollinations AI.
No database. All state lives in st.session_state.
"""

import json
import re
import urllib.parse
from datetime import datetime

import streamlit as st
from PIL import Image

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except Exception:  # SDK missing -> app still runs in demo mode
    GENAI_AVAILABLE = False

# ----------------------------------------------------------------------------
# Page config & styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Construction Intelligence Platform",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
.block-container {padding-top: 1.5rem; max-width: 1250px;}
.hero {
    background: linear-gradient(120deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    padding: 1.6rem 2rem; border-radius: 16px; color: #fff; margin-bottom: 1.2rem;
}
.hero h1 {margin: 0; font-size: 2rem; color: #fff;}
.hero p {margin: .3rem 0 0; opacity: .85;}
.mcard {
    background: #ffffff; border: 1px solid #e5e9f0; border-left: 5px solid #2c5364;
    border-radius: 12px; padding: 1rem 1.2rem; box-shadow: 0 2px 8px rgba(0,0,0,.05);
    color: #1b2733;
}
.mcard .lbl {font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; color: #6b7785;}
.mcard .val {font-size: 1.7rem; font-weight: 700; line-height: 1.2;}
.mcard .sub {font-size: .8rem; color: #6b7785;}
.pill {
    display: inline-block; padding: .22rem .75rem; border-radius: 999px;
    font-size: .8rem; font-weight: 600; color: #fff;
}
.pill.green {background: #1e9e5a;} .pill.amber {background: #e6a100;}
.pill.orange {background: #e8710a;} .pill.red {background: #d93025;}
.pill.dark {background: #7a0c0c;} .pill.grey {background: #6b7785;}
.hazard {
    background: #fafbfc; border: 1px solid #e5e9f0; border-radius: 10px;
    padding: .8rem 1rem; margin-bottom: .6rem; color: #1b2733;
}
.alert-box {border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: .6rem; color: #1b2733;}
.demo-banner {
    background: #fff4e5; border: 1px solid #ffcc80; color: #7a4a00;
    padding: .6rem 1rem; border-radius: 10px; margin-bottom: 1rem; font-size: .9rem;
}
.disclaimer {font-size: .78rem; color: #6b7785; margin-top: .8rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Generic helpers
# ----------------------------------------------------------------------------
SEVERITY_COLOR = {"low": "green", "medium": "amber", "high": "orange", "critical": "red"}
DEFAULT_MODEL = "gemini-flash-latest"


def pill(text: str, color: str) -> str:
    return f'<span class="pill {color}">{text}</span>'


def metric_card(label: str, value: str, sub: str = "", color: str = "#2c5364") -> str:
    return (
        f'<div class="mcard" style="border-left-color:{color}">'
        f'<div class="lbl">{label}</div><div class="val">{value}</div>'
        f'<div class="sub">{sub}</div></div>'
    )


def format_inr(amount: float) -> str:
    """Format a rupee amount as Crores / Lakhs."""
    if amount >= 1e7:
        return f"₹ {amount / 1e7:,.2f} Cr"
    if amount >= 1e5:
        return f"₹ {amount / 1e5:,.2f} Lakhs"
    return f"₹ {amount:,.0f}"


def load_image(uploaded) -> Image.Image:
    """Open an uploaded file, normalise to RGB and cap size for API efficiency."""
    img = Image.open(uploaded)
    img = img.convert("RGB")
    img.thumbnail((1600, 1600))
    return img


def parse_json(text: str) -> dict:
    """Parse model output as JSON, tolerating markdown fences and stray text."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def clamp(value, lo, hi, default):
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return default


def get_api_key() -> str:
    key = st.session_state.get("api_key_input", "").strip()
    if key:
        return key
    try:  # optional: Streamlit Cloud secrets
        return st.secrets.get("GEMINI_API_KEY", "").strip()
    except Exception:
        return ""


def live_mode() -> bool:
    return GENAI_AVAILABLE and bool(get_api_key())


def gemini_json(prompt: str, image: Image.Image) -> dict:
    """Multimodal call returning parsed JSON."""
    client = genai.Client(api_key=get_api_key())
    resp = client.models.generate_content(
        model=st.session_state.get("model_name", DEFAULT_MODEL),
        contents=[prompt, image],
        config=types.GenerateContentConfig(
            response_mime_type="application/json", temperature=0.2
        ),
    )
    return parse_json(resp.text)


def gemini_chat(system_prompt: str, history: list) -> str:
    """Multi-turn chat: the transcript is replayed each turn (stateless, no DB)."""
    client = genai.Client(api_key=get_api_key())
    transcript = "\n".join(
        f"{'Engineer' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in history
    )
    resp = client.models.generate_content(
        model=st.session_state.get("model_name", DEFAULT_MODEL),
        contents=transcript + "\nAssistant:",
        config=types.GenerateContentConfig(
            system_instruction=system_prompt, temperature=0.4
        ),
    )
    return (resp.text or "").strip()


def friendly_error(exc: Exception) -> str:
    msg = str(exc)
    if "API key" in msg or "API_KEY" in msg or "403" in msg or "401" in msg:
        return "The Gemini API key was rejected. Please check it in the sidebar."
    if "429" in msg or "quota" in msg.lower():
        return "Gemini rate limit or quota reached. Wait a moment and retry."
    if "404" in msg:
        return "Model not found. Pick another model name in the sidebar."
    return f"Analysis failed: {msg[:300]}"


def demo_banner():
    if not live_mode():
        st.markdown(
            '<div class="demo-banner">🧪 <b>Demonstration mode</b> — no Gemini API key '
            "supplied, so results below are sample data. Add a key in the sidebar for "
            "real image analysis.</div>",
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------
SAFETY_PROMPT = """You are a certified construction safety officer (OSHA 1926 and Indian
BOCW / IS 7969 aware). Analyse this construction site photograph for PPE compliance and
site hazards. Where concrete or structural-quality observations are relevant (formwork,
curing, reinforcement cover, honeycombing), reference IS 456:2000. Only report what is
visible; use "Not Visible" when a PPE item cannot be assessed. Do not invent workers.

Return ONLY valid JSON with this exact schema:
{
  "safety_score": <integer 0-100>,
  "overall_status": "Compliant" | "Needs Improvement" | "Non-Compliant" | "Critical",
  "workers_detected": <integer>,
  "summary": "<2-3 sentence assessment>",
  "ppe_detection": {
    "helmet": {"status": "Present" | "Missing" | "Not Visible", "remarks": "<short>"},
    "high_vis_vest": {"status": "...", "remarks": "<short>"},
    "steel_toe_boots": {"status": "...", "remarks": "<short>"},
    "safety_harness": {"status": "...", "remarks": "<short>"}
  },
  "hazards": [
    {"hazard": "<name>", "severity": "Low" | "Medium" | "High" | "Critical",
     "location": "<where in image>", "remedy": "<corrective action>",
     "standard_reference": "<OSHA/IS clause or 'General best practice'>"}
  ]
}"""

CRACK_PROMPT = """You are a senior structural engineer specialising in concrete and masonry
defect diagnosis (IS 456:2000, IS 13935, IS 15988, IS 516 aware). Analyse the photograph of
a concrete/wall surface. Classify the defect into exactly one of: Hairline, Settlement,
Structural Shear, Thermal Expansion, Spalling. Estimate crack width in millimetres from visual
cues (state uncertainty in the confidence field). If the image does not show a defect, say so
in summary and use the closest class with low confidence.

Return ONLY valid JSON with this exact schema:
{
  "defect_type": "Hairline" | "Settlement" | "Structural Shear" | "Thermal Expansion" | "Spalling",
  "confidence": <integer 0-100>,
  "estimated_width_mm": <number>,
  "orientation": "<vertical / horizontal / diagonal / map-pattern / etc>",
  "severity": "Low" | "Medium" | "High" | "Critical",
  "urgency": "Monitor" | "Repair within 3 months" | "Repair within 2 weeks" | "Immediate action",
  "summary": "<2-3 sentences>",
  "root_causes": ["<cause>", "..."],
  "repairs": [{"method": "<e.g. Epoxy injection>", "description": "<how/when to apply>"}],
  "monitoring_plan": "<e.g. crack gauge / tell-tales, interval>"
}"""

ASSISTANT_SYSTEM = """You are 'SiteMate', an expert construction engineering assistant for
Indian building sites. Give practical, concise, safety-first answers on civil/structural
engineering, concrete technology, scheduling, material estimation, site safety and codes
(IS 456, IS 800, IS 1893, NBC 2016, OSHA). Use metric units and INR where relevant. Show
working for calculations. If a question needs a licensed structural engineer's sign-off, say
so. Keep answers under 200 words unless asked for detail.
Current site weather condition: {weather}."""

# ----------------------------------------------------------------------------
# Demo-mode sample data
# ----------------------------------------------------------------------------
DEMO_SAFETY = {
    "safety_score": 62,
    "overall_status": "Needs Improvement",
    "workers_detected": 3,
    "summary": "Sample result: most workers wear helmets and vests, but one worker is "
               "at height without a harness and the scaffold edge has no toe board.",
    "ppe_detection": {
        "helmet": {"status": "Present", "remarks": "3 of 3 workers wearing helmets"},
        "high_vis_vest": {"status": "Present", "remarks": "2 of 3 workers; 1 missing vest"},
        "steel_toe_boots": {"status": "Not Visible", "remarks": "Footwear obscured by materials"},
        "safety_harness": {"status": "Missing", "remarks": "Worker on upper deck unprotected"},
    },
    "hazards": [
        {"hazard": "Work at height without fall protection", "severity": "Critical",
         "location": "Upper deck, centre-right",
         "remedy": "Stop work; provide full-body harness anchored to a rated lifeline.",
         "standard_reference": "OSHA 1926.501(b)"},
        {"hazard": "Scaffold edge without toe board / guardrail", "severity": "High",
         "location": "Left scaffold bay",
         "remedy": "Install guardrails at 1 m and toe boards of 150 mm minimum.",
         "standard_reference": "OSHA 1926.451(g)"},
        {"hazard": "Exposed reinforcement bar ends", "severity": "Medium",
         "location": "Foreground columns",
         "remedy": "Fit mushroom rebar caps on all protruding bars.",
         "standard_reference": "OSHA 1926.701(b)"},
        {"hazard": "Loose material stacking near walkway", "severity": "Low",
         "location": "Bottom-left",
         "remedy": "Relocate stacks and keep a 1.2 m clear access path.",
         "standard_reference": "General best practice"},
    ],
}

DEMO_CRACK = {
    "defect_type": "Settlement",
    "confidence": 78,
    "estimated_width_mm": 3.5,
    "orientation": "diagonal, stepped along mortar joints",
    "severity": "High",
    "urgency": "Repair within 2 weeks",
    "summary": "Sample result: a diagonal stepped crack of about 3-4 mm suggests "
               "differential foundation settlement rather than shrinkage.",
    "root_causes": [
        "Differential settlement of footing due to soft or poorly compacted soil",
        "Water ingress softening the sub-grade near the foundation",
        "Inadequate foundation depth or missing plinth beam tie",
    ],
    "repairs": [
        {"method": "Crack-width monitoring first",
         "description": "Install tell-tales for 4 weeks to confirm movement has stabilised."},
        {"method": "Epoxy injection",
         "description": "Once stable, inject low-viscosity epoxy to restore monolithic action."},
        {"method": "Structural re-grouting / underpinning",
         "description": "If movement continues, grout the sub-grade or underpin the footing "
                        "to a design by a structural engineer."},
    ],
    "monitoring_plan": "Fix glass tell-tales and review weekly for 4 weeks, then monthly for 6 months.",
}

# ----------------------------------------------------------------------------
# Valuation model (Tab 3) - transparent, editable assumptions
# ----------------------------------------------------------------------------
# cost_per_sqft = construction cost (INR, Tier-1 baseline, structure + finishing)
# market_per_sqft = market value of completed built-up area, excluding land (Tier-1)
# steel_kg / cement_bags = per sqft of built-up area
STRUCTURE_TYPES = {
    "RCC Residential (Apartment / House)": {"cost": 2000, "market": 3600, "steel": 4.0, "cement": 0.40},
    "RCC Commercial / Office": {"cost": 2600, "market": 4800, "steel": 4.5, "cement": 0.45},
    "Premium Villa": {"cost": 3200, "market": 5800, "steel": 4.2, "cement": 0.42},
    "Load-bearing Masonry (G+1)": {"cost": 1500, "market": 2600, "steel": 2.0, "cement": 0.35},
    "Industrial Shed (PEB)": {"cost": 1700, "market": 2400, "steel": 7.5, "cement": 0.20},
}
TIERS = {
    "Tier 1 (Mumbai, Delhi NCR, Bengaluru)": {"cost": 1.00, "market": 1.00, "examples": "metro city"},
    "Tier 2 (Indore, Pune, Jaipur, Lucknow)": {"cost": 0.82, "market": 0.70, "examples": "large city"},
    "Tier 3 (smaller towns & district HQs)": {"cost": 0.68, "market": 0.48, "examples": "small town"},
}
MILESTONES = [
    ("Site prep & foundation", 15),
    ("Plinth & substructure", 30),
    ("Superstructure (RCC frame)", 60),
    ("Masonry & roofing", 75),
    ("MEP & plastering", 90),
    ("Finishing & handover", 100),
]

# ----------------------------------------------------------------------------
# Weather advisory data (Tab 4)
# ----------------------------------------------------------------------------
WEATHER = {
    "Normal": {
        "icon": "☀️", "color": "#e8f5e9", "status": "Full operations", "level": "green",
        "alerts": [
            "Standard operations permitted; maintain toolbox talk and PPE checks.",
            "Good window for concrete pours and crane lifts - verify pre-pour checklist.",
            "Keep potable water available and rotate workers for rest breaks.",
        ],
    },
    "High Wind": {
        "icon": "💨", "color": "#fff3e0", "status": "Restricted operations", "level": "orange",
        "alerts": [
            "🛑 SUSPEND crane and hoist operations when sustained wind exceeds the "
            "manufacturer's limit (commonly ~36 km/h / 10 m/s); lower and secure jibs.",
            "Stop work on scaffolds, facade and roof edges; evacuate open decks.",
            "Tie down sheets, formwork panels, bamboo/steel stacks and loose materials.",
            "Barricade areas beneath tall structures to guard against falling objects.",
        ],
    },
    "Heavy Rain": {
        "icon": "🌧️", "color": "#e3f2fd", "status": "Hazardous - limited work", "level": "red",
        "alerts": [
            "⚡ ELECTRICAL HAZARD: de-energise temporary supplies, check RCCB/ELCB, keep "
            "cables off the ground and out of puddles, no wet-hand tool use.",
            "Postpone concrete pours; cover fresh concrete with tarpaulin to prevent "
            "washout and surface damage.",
            "Pump out excavations and inspect trench walls for collapse risk before entry.",
            "Wet scaffold boards and ladders are slip hazards - inspect before use.",
            "Protect cement and other bagged materials on raised, covered platforms.",
        ],
    },
    "Extreme Heat": {
        "icon": "🔥", "color": "#fff8e1", "status": "Heat-stress controls on", "level": "amber",
        "alerts": [
            "🧱 CONCRETE CURING: begin wet curing immediately, use wet burlap or "
            "curing compound, pour early morning or evening, and keep fresh concrete "
            "temperature below ~35 °C (IS 456 hot-weather concreting).",
            "Work-rest cycles: 15 minutes of rest per hour at high heat index; shaded rest area.",
            "Provide ORS and drinking water; watch for heat exhaustion and heat stroke.",
            "Avoid heavy tasks 12:00-16:00; shift hard work to cooler hours.",
            "Steel and formwork surfaces can burn skin - provide gloves.",
        ],
    },
}

KB_DEMO = [
    (("cure", "curing"), "Cure concrete for at least 7 days for OPC (10 days where mineral admixtures "
     "are used, per IS 456 cl. 13.5). Keep surfaces continuously wet, and extend curing in hot or windy weather."),
    (("cement", "bags"), "Rule of thumb: about 0.4 bags (50 kg) of cement per sqft of built-up area for "
     "RCC residential. For 1 m³ of M20 (1:1.5:3) concrete you need roughly 8 bags."),
    (("steel", "rebar"), "Typical steel consumption: 3.5-4.5 kg/sqft for RCC residential. "
     "Always follow the structural drawings, since the rule of thumb is for budgeting only."),
    (("crack",), "Classify first: hairline (<0.3 mm) is usually shrinkage, diagonal stepped cracks suggest settlement, "
     "and 45° cracks near supports suggest shear. Monitor width with tell-tales before repairing."),
    (("ppe", "helmet", "harness"), "Mandatory PPE: helmet, high-vis vest, safety shoes, and a harness for any work above "
     "1.8 m (OSHA) / 2 m (BOCW). Inspect harnesses before every use."),
    (("slump",), "Typical slump: 25-75 mm for footings and slabs with vibrators, 75-100 mm for columns and "
     "beams, up to 100-150 mm for pumped concrete."),
]


def demo_chat_reply(question: str, weather: str) -> str:
    q = question.lower()
    for keys, answer in KB_DEMO:
        if any(k in q for k in keys):
            return answer + "\n\n*(Demo mode - add a Gemini key for full conversational answers.)*"
    return (f"Demo mode: I can answer basics on curing, cement/steel quantities, cracks, PPE and slump. "
            f"Today's weather setting is **{weather}**. Add a Gemini API key in the sidebar for open-ended help.")


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🏗️ Control Panel")
        st.text_input(
            "Google Gemini API key",
            type="password",
            key="api_key_input",
            help="Get a free key at aistudio.google.com. The key lives only in this "
                 "browser session and is never stored.",
            placeholder="AIza...",
        )
        st.text_input("Model", value=DEFAULT_MODEL, key="model_name")

        if not GENAI_AVAILABLE:
            st.error("`google-genai` is not installed. Running in demo mode.")
        elif live_mode():
            st.success("🟢 Live mode: Gemini connected")
        else:
            st.warning("🟠 Demo mode: no API key")

        st.divider()
        st.markdown("### 🧭 Feature navigation")
        st.markdown(
            "- 🦺 **Safety & PPE Audit**: photo → OSHA score\n"
            "- 🧱 **Crack Diagnostics**: defect class & repairs\n"
            "- 💰 **Planning & Valuation**: budget, capex, assets\n"
            "- 🌦️ **Daily Assistant**: weather alerts & chatbot"
        )
        st.caption("Use the tabs on the main page to switch modules.")

        st.divider()
        with st.expander("🏛️ Architecture overview"):
            st.markdown(
                """
```
 Streamlit UI (4 tabs)
      │
 ┌────┴──────────────────────────┐
 │ Image tabs   Calc tab   Chat  │
 │ (PIL → prompt)  (pure   (multi│
 │      │          Python)  turn)│
 └──────┼──────────────────┼─────┘
        ▼                  ▼
  Gemini multimodal   Gemini text
  → strict JSON       + system prompt
        ▼
  Validation → UI cards → JSON export
```
- **No database**: state is `st.session_state`
- **Fallback**: demo data without a key
- **Rendering**: Pollinations AI (URL based)
"""
            )
        st.caption("Skill Wallet · Capstone Project 1")


# ----------------------------------------------------------------------------
# Tab 1 - Safety & PPE
# ----------------------------------------------------------------------------
def normalise_safety(d: dict) -> dict:
    d = d if isinstance(d, dict) else {}
    d["safety_score"] = int(clamp(d.get("safety_score"), 0, 100, 0))
    d.setdefault("overall_status", "Needs Improvement")
    d.setdefault("workers_detected", 0)
    d.setdefault("summary", "")
    ppe = d.get("ppe_detection") or {}
    for item in ("helmet", "high_vis_vest", "steel_toe_boots", "safety_harness"):
        ppe.setdefault(item, {"status": "Not Visible", "remarks": ""})
    d["ppe_detection"] = ppe
    d["hazards"] = d.get("hazards") or []
    return d


def render_safety(report: dict):
    score = report["safety_score"]
    status = report["overall_status"]
    score_color = "#1e9e5a" if score >= 80 else "#e6a100" if score >= 60 else "#d93025"
    status_color = {"compliant": "green", "needs improvement": "amber",
                    "non-compliant": "red", "critical": "dark"}.get(status.lower(), "grey")

    hazards = report["hazards"]
    crit = sum(1 for h in hazards if str(h.get("severity", "")).lower() in ("critical", "high"))

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Safety score", f"{score}/100", "OSHA-based", score_color), unsafe_allow_html=True)
    c2.markdown(metric_card("Overall status", pill(status, status_color), "compliance rating"), unsafe_allow_html=True)
    c3.markdown(metric_card("Workers detected", str(report["workers_detected"]), "in frame"), unsafe_allow_html=True)
    c4.markdown(metric_card("High / critical hazards", str(crit), f"{len(hazards)} total", "#d93025" if crit else "#1e9e5a"),
                unsafe_allow_html=True)

    if report["summary"]:
        st.info(report["summary"])

    st.markdown("#### 🦺 PPE checklist")
    labels = {"helmet": "⛑️ Helmet", "high_vis_vest": "🦺 High-vis vest",
              "steel_toe_boots": "🥾 Steel-toe boots", "safety_harness": "🪢 Safety harness"}
    cols = st.columns(4)
    for col, (key, label) in zip(cols, labels.items()):
        item = report["ppe_detection"][key]
        s = str(item.get("status", "Not Visible"))
        color = {"present": "green", "missing": "red"}.get(s.lower(), "grey")
        col.markdown(
            f'<div class="hazard"><b>{label}</b><br>{pill(s, color)}'
            f'<br><small>{item.get("remarks", "")}</small></div>',
            unsafe_allow_html=True,
        )

    st.markdown("#### ⚠️ Identified hazards")
    if not hazards:
        st.success("No hazards identified in this image.")
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    for h in sorted(hazards, key=lambda x: order.get(str(x.get("severity", "")).lower(), 9)):
        sev = str(h.get("severity", "Low"))
        st.markdown(
            f'<div class="hazard">{pill(sev, SEVERITY_COLOR.get(sev.lower(), "grey"))} '
            f'&nbsp;<b>{h.get("hazard", "Hazard")}</b><br>'
            f'<small>📍 {h.get("location", "n/a")} &nbsp;|&nbsp; 📘 {h.get("standard_reference", "n/a")}</small><br>'
            f'🔧 <b>Remedy:</b> {h.get("remedy", "n/a")}</div>',
            unsafe_allow_html=True,
        )

    st.download_button(
        "⬇️ Download JSON report",
        data=json.dumps(report, indent=2),
        file_name=f"safety_report_{datetime.now():%Y%m%d_%H%M%S}.json",
        mime="application/json",
    )


def tab_safety():
    st.subheader("🦺 Site Safety & PPE Audit")
    st.caption("Upload a construction site photo. Gemini audits PPE compliance and hazards "
               "against OSHA and relevant IS standards.")
    demo_banner()
    up = st.file_uploader("Site photo (JPG / PNG)", type=["jpg", "jpeg", "png"], key="safety_up")
    if not up:
        return
    try:
        img = load_image(up)
    except Exception:
        st.error("Could not read that image. Please upload a valid JPG or PNG.")
        return

    left, right = st.columns([1, 2])
    left.image(img, caption="Uploaded site photo", width="stretch")
    with right:
        if st.button("🔍 Run safety audit", type="primary", key="safety_btn"):
            with st.spinner("Analysing site photo..."):
                try:
                    raw = gemini_json(SAFETY_PROMPT, img) if live_mode() else DEMO_SAFETY
                    st.session_state["safety_report"] = normalise_safety(raw)
                except Exception as e:
                    st.error(friendly_error(e))
        if "safety_report" in st.session_state:
            render_safety(st.session_state["safety_report"])
    st.markdown('<div class="disclaimer">AI-assisted screening only. A qualified safety officer '
                'must verify findings on site.</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Tab 2 - Crack diagnostics
# ----------------------------------------------------------------------------
def normalise_crack(d: dict) -> dict:
    d = d if isinstance(d, dict) else {}
    d.setdefault("defect_type", "Hairline")
    d["confidence"] = int(clamp(d.get("confidence"), 0, 100, 0))
    d["estimated_width_mm"] = round(clamp(d.get("estimated_width_mm"), 0, 500, 0.0), 2)
    d.setdefault("orientation", "n/a")
    d.setdefault("severity", "Medium")
    d.setdefault("urgency", "Monitor")
    d.setdefault("summary", "")
    d["root_causes"] = d.get("root_causes") or []
    d["repairs"] = d.get("repairs") or []
    d.setdefault("monitoring_plan", "")
    return d


def render_crack(r: dict):
    sev = str(r["severity"])
    sev_color = SEVERITY_COLOR.get(sev.lower(), "grey")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Defect type", r["defect_type"], f'{r["confidence"]}% confidence'), unsafe_allow_html=True)
    c2.markdown(metric_card("Est. crack width", f'{r["estimated_width_mm"]} mm', r["orientation"]), unsafe_allow_html=True)
    c3.markdown(metric_card("Severity", pill(sev, sev_color), "structural risk"), unsafe_allow_html=True)
    c4.markdown(metric_card("Urgency", f'<span style="font-size:1.05rem">{r["urgency"]}</span>', "recommended timeline"),
                unsafe_allow_html=True)

    if r["summary"]:
        st.info(r["summary"])

    a, b = st.columns(2)
    with a:
        st.markdown("#### 🔎 Probable root causes")
        for c in r["root_causes"]:
            st.markdown(f"- {c}")
    with b:
        st.markdown("#### 🛠️ Recommended engineering repairs")
        for rep in r["repairs"]:
            if isinstance(rep, dict):
                st.markdown(f"**{rep.get('method', 'Repair')}**: {rep.get('description', '')}")
            else:
                st.markdown(f"- {rep}")
    if r["monitoring_plan"]:
        st.markdown(f"**📈 Monitoring plan:** {r['monitoring_plan']}")

    st.download_button(
        "⬇️ Download JSON report",
        data=json.dumps(r, indent=2),
        file_name=f"crack_report_{datetime.now():%Y%m%d_%H%M%S}.json",
        mime="application/json",
    )


def tab_crack():
    st.subheader("🧱 Crack Diagnostics & Structural Integrity")
    st.caption("Upload a concrete or wall surface photo to classify the defect and get repair guidance.")
    demo_banner()
    up = st.file_uploader("Surface photo (JPG / PNG)", type=["jpg", "jpeg", "png"], key="crack_up")
    if not up:
        return
    try:
        img = load_image(up)
    except Exception:
        st.error("Could not read that image. Please upload a valid JPG or PNG.")
        return

    left, right = st.columns([1, 2])
    left.image(img, caption="Uploaded surface", width="stretch")
    with right:
        if st.button("🔍 Diagnose defect", type="primary", key="crack_btn"):
            with st.spinner("Analysing surface..."):
                try:
                    raw = gemini_json(CRACK_PROMPT, img) if live_mode() else DEMO_CRACK
                    st.session_state["crack_report"] = normalise_crack(raw)
                except Exception as e:
                    st.error(friendly_error(e))
        if "crack_report" in st.session_state:
            render_crack(st.session_state["crack_report"])
    st.markdown('<div class="disclaimer">Preliminary AI assessment from a photograph. Crack width from '
                'images is approximate. Have a licensed structural engineer inspect before any repair.</div>',
                unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Tab 3 - Planning & valuation
# ----------------------------------------------------------------------------
def compute_project(structure: str, sqft: float, tier: str, completion: int) -> dict:
    s, t = STRUCTURE_TYPES[structure], TIERS[tier]
    frac = completion / 100
    budget = sqft * s["cost"] * t["cost"]
    full_value = sqft * s["market"] * t["market"]
    steel_total = sqft * s["steel"]
    cement_total = sqft * s["cement"]
    return {
        "budget": budget,
        "capex": budget * frac,
        "remaining": budget * (1 - frac),
        "full_value": full_value,
        "asset_value": full_value * frac,
        "steel_total_t": steel_total / 1000,
        "steel_used_t": steel_total * frac / 1000,
        "cement_total": cement_total,
        "cement_used": cement_total * frac,
        "cost_psf": s["cost"] * t["cost"],
    }


def build_render_prompt(structure: str, sqft: float, tier: str, style: str, completion: int) -> str:
    stage = "completed" if completion >= 100 else f"under construction, {completion}% complete"
    return (f"Photorealistic architectural 3D rendering of a {structure.split('(')[0].strip().lower()}, "
            f"about {int(sqft)} sq ft built-up area, {style} style, Indian {tier.split('(')[0].strip().lower()} "
            f"setting, {stage}, golden hour lighting, landscaped surroundings, highly detailed")


def tab_planning():
    st.subheader("💰 Planning & Real Estate Valuation")
    st.caption("Estimate budget, capital expenditure, asset value and material needs for a project.")

    c1, c2 = st.columns(2)
    structure = c1.selectbox("Structure type", list(STRUCTURE_TYPES.keys()))
    tier = c2.selectbox("Location tier", list(TIERS.keys()), index=1)
    c3, c4 = st.columns(2)
    sqft = c3.number_input("Built-up area (sq ft)", min_value=100, max_value=1_000_000, value=2000, step=100)
    style = c4.selectbox("Architectural style (for rendering)",
                         ["Modern contemporary", "Minimalist", "Traditional Indian", "Colonial", "Industrial"])
    completion = st.slider("Completion percentage", 0, 100, 40, 5)

    r = compute_project(structure, sqft, tier, completion)

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(metric_card("Total project budget", format_inr(r["budget"]), f'@ ₹{r["cost_psf"]:,.0f}/sq ft'),
                unsafe_allow_html=True)
    m2.markdown(metric_card("Capex incurred", format_inr(r["capex"]), f'{completion}% of budget', "#e8710a"),
                unsafe_allow_html=True)
    m3.markdown(metric_card("Current asset valuation", format_inr(r["asset_value"]),
                            f'full value {format_inr(r["full_value"])}', "#1e9e5a"), unsafe_allow_html=True)
    m4.markdown(metric_card("Remaining spend", format_inr(r["remaining"]), "to completion", "#6b7785"),
                unsafe_allow_html=True)

    st.markdown("#### 🧱 Material requirements")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Steel (total)", f'{r["steel_total_t"]:.2f} tonnes')
    k2.metric("Steel consumed", f'{r["steel_used_t"]:.2f} tonnes')
    k3.metric("Cement (total)", f'{r["cement_total"]:,.0f} bags')
    k4.metric("Cement consumed", f'{r["cement_used"]:,.0f} bags')

    st.markdown("#### 🚧 Milestone progress")
    st.progress(completion / 100, text=f"Project {completion}% complete")
    cols = st.columns(len(MILESTONES))
    prev = 0
    for col, (name, upto) in zip(cols, MILESTONES):
        done = completion >= upto
        active = prev <= completion < upto
        icon = "✅" if done else "🔨" if active else "⏳"
        col.markdown(f"{icon}<br><small><b>{name}</b><br>{prev}-{upto}%</small>", unsafe_allow_html=True)
        prev = upto

    with st.expander("ℹ️ Assumptions & method"):
        st.markdown(
            "- Budget = built-up area × construction cost/sq ft × location-tier factor.\n"
            "- Asset valuation = market value of the completed structure (land excluded) × completion %.\n"
            "- Steel/cement use rule-of-thumb kg and bags per sq ft by structure type.\n"
            "- Rates are indicative planning figures. Replace with local schedule of rates (SOR) or a "
            "quantity surveyor's estimate for real decisions. Edit `STRUCTURE_TYPES` and `TIERS` in the code to calibrate."
        )

    st.markdown("#### 🎨 Conceptual 3D architectural rendering")
    prompt = build_render_prompt(structure, sqft, tier, style, completion)
    prompt = st.text_area("Rendering prompt (editable)", value=prompt, height=90)
    if st.button("🖼️ Generate rendering (Pollinations AI)", key="render_btn"):
        st.session_state["render_url"] = (
            "https://image.pollinations.ai/prompt/" + urllib.parse.quote(prompt)
            + "?width=1024&height=640&nologo=true&seed=" + str(int(datetime.now().timestamp()) % 100000)
        )
    if st.session_state.get("render_url"):
        st.image(st.session_state["render_url"], caption="AI concept render (illustrative only)",
                 width="stretch")

    summary = {"inputs": {"structure": structure, "tier": tier, "sqft": sqft, "completion_pct": completion},
               "outputs": {k: round(v, 2) for k, v in r.items()}}
    st.download_button("⬇️ Download estimate (JSON)", data=json.dumps(summary, indent=2),
                       file_name="project_estimate.json", mime="application/json")


# ----------------------------------------------------------------------------
# Tab 4 - Daily assistant
# ----------------------------------------------------------------------------
def tab_assistant():
    st.subheader("🌦️ Daily Site Assistant & Weather-Aware Advisory")
    weather = st.radio("Today's site weather", list(WEATHER.keys()), horizontal=True, key="weather")
    w = WEATHER[weather]

    st.markdown(
        f'<div class="alert-box" style="background:{w["color"]}">'
        f'<span style="font-size:1.6rem">{w["icon"]}</span> &nbsp;<b>{weather}</b> &nbsp;'
        f'{pill(w["status"], w["level"])}</div>',
        unsafe_allow_html=True,
    )
    for a in w["alerts"]:
        st.markdown(f'<div class="alert-box" style="background:#fafbfc;border:1px solid #e5e9f0">{a}</div>',
                    unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 💬 SiteMate: construction engineering assistant")
    demo_banner()

    if "chat" not in st.session_state:
        st.session_state["chat"] = [{
            "role": "assistant",
            "content": "Hi, I'm SiteMate 👷. Ask me about concrete curing, material quantities, "
                       "scheduling, codes or site safety.",
        }]

    top = st.columns([6, 1])[1]
    if top.button("🗑️ Clear", key="clear_chat"):
        del st.session_state["chat"]
        st.rerun()

    for m in st.session_state["chat"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if question := st.chat_input("Ask a construction question..."):
        st.session_state["chat"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if live_mode():
                        reply = gemini_chat(ASSISTANT_SYSTEM.format(weather=weather),
                                            st.session_state["chat"][-12:])
                    else:
                        reply = demo_chat_reply(question, weather)
                except Exception as e:
                    reply = f"⚠️ {friendly_error(e)}"
            st.markdown(reply or "_No response received._")
        st.session_state["chat"].append({"role": "assistant", "content": reply})

        
    st.divider()
    st.markdown("#### 📋 Daily Construction Action Plan")

    with st.form("daily_site_plan_form"):
        activity = st.selectbox(
            "Today's construction activity",
            [
                "Concrete pouring",
                "Brickwork",
                "Plastering",
                "Steel reinforcement",
                "Excavation",
                "Painting",
                "General site work",
            ],
        )

        labour_count = st.number_input(
            "Available workers",
            min_value=1,
            max_value=500,
            value=10,
            step=1,
        )

        site_condition = st.selectbox(
            "Site conditions",
            ["Normal", "Hot weather", "Rainy", "Windy", "Very hot and dry"],
        )

        submitted = st.form_submit_button("Generate Daily Plan")

    if submitted:
        prompt = f"""
        Create a practical daily construction action plan for an Indian building site.

        Activity: {activity}
        Available workers: {labour_count}
        Site conditions: {site_condition}
        Weather advisory: {weather}

        Return valid JSON with these keys:
        - tasks: list of daily tasks
        - materials_equipment: list of required materials and equipment
        - safety_checks: list of safety checks
        - risk_alerts: list of potential risks and precautions

        Keep recommendations concise and actionable.
        Do not invent exact material quantities without sufficient information.
        Treat structural and safety recommendations as preliminary guidance.
        """

        try:
            with st.spinner("Preparing your daily site plan..."):
                if live_mode():
                    client = genai.Client(api_key=get_api_key())
                    response = client.models.generate_content(
                        model=st.session_state.get("model_name", DEFAULT_MODEL),
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.2,
                        ),
                    )
                    plan = parse_json(response.text)
                else:
                    plan = {
                        "tasks": [
                            f"Brief workers on today's {activity.lower()} work.",
                            "Assign responsibilities and confirm the work sequence.",
                            "Inspect completed work before proceeding.",
                        ],
                        "materials_equipment": [
                            "Verify required materials and equipment with the site supervisor."
                        ],
                        "safety_checks": [
                            "Check PPE, access routes, tools and equipment before work."
                        ],
                        "risk_alerts": [
                            f"Review the risks associated with {site_condition.lower()} conditions."
                        ],
                    }

            st.markdown("### Today's Action Plan")

            sections = [
                ("tasks", "Daily Tasks"),
                ("materials_equipment", "Materials & Equipment"),
                ("safety_checks", "Safety Checks"),
                ("risk_alerts", "Risk Alerts"),
            ]

            for key, heading in sections:
                st.markdown(f"#### {heading}")
                items = plan.get(key, [])
                if isinstance(items, list) and items:
                    for item in items:
                        st.markdown(f"- {item}")
                else:
                    st.info("No items were returned for this section.")

            st.download_button(
                "Download Daily Plan (JSON)",
                data=__import__("json").dumps(plan, indent=2),
                file_name="daily_site_plan.json",
                mime="application/json",
            )

        except Exception as e:
            st.error(f"Could not generate the daily plan: {friendly_error(e)}")





# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    render_sidebar()
    st.markdown(
        '<div class="hero"><h1>🏗️ Multimodal Construction Intelligence Platform</h1>'
        "<p>AI-powered site safety, structural diagnostics, project valuation and daily advisory · "
        "Skill Wallet Capstone Project 1</p></div>",
        unsafe_allow_html=True,
    )
    t1, t2, t3, t4 = st.tabs([
        "🦺 Safety & PPE Audit", "🧱 Crack Diagnostics",
        "💰 Planning & Valuation", "🌦️ Daily Assistant",
    ])
    with t1:
        tab_safety()
    with t2:
        tab_crack()
    with t3:
        tab_planning()
    with t4:
        tab_assistant()


if __name__ == "__main__":
    main()
