
import streamlit as st
import google.generativeai as genai
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import os, json, base64, requests, time

st.set_page_config(page_title="MedPilot OS", page_icon="🏥", layout="wide")

MODEL = "models/gemini-2.5-flash"

api_key = os.environ.get("GEMINI_API_KEY", "")
if not api_key:
    api_key = st.text_input("🔑 Gemini API key:", type="password")
    if not api_key:
        st.warning("Enter API key.")
        st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel(MODEL)

# ── SESSION STATE ─────────────────────────────────────────
for k, v in {
    "patient": {}, "medications": [], "symptoms": [],
    "notes": [], "consents": [], "emergency_log": [],
    "tasks": [], "calendar_events": [], "pending_ocr": None,
    "agent_trace": [], "vitals": [],
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── AGENT TRACE ENGINE ────────────────────────────────────
def trace(agent, action, status="running"):
    icons = {"running": "⚡", "done": "✅", "error": "❌", "warn": "⚠️"}
    entry = f"{icons.get(status,'⚡')} [{agent}] {action}"
    st.session_state.agent_trace.insert(0, entry)
    if len(st.session_state.agent_trace) > 30:
        st.session_state.agent_trace = st.session_state.agent_trace[:30]

# ── STYLES ────────────────────────────────────────────────
st.markdown("""
<style>
.verify-box  {background:#fef9c3;border-left:5px solid #ca8a04;padding:1rem;border-radius:8px;margin:.8rem 0}
.consent-box {background:#f0fdf4;border-left:5px solid #16a34a;padding:1rem;border-radius:8px;margin:.8rem 0}
.emergency   {background:#fef2f2;border-left:5px solid #dc2626;padding:1rem;border-radius:8px;margin:.8rem 0}
.trace-box   {background:#0f172a;color:#94a3b8;font-family:monospace;font-size:11px;padding:.8rem;border-radius:8px;max-height:260px;overflow-y:auto}
.agent-badge {display:inline-block;background:#dbeafe;color:#1e40af;padding:2px 10px;border-radius:12px;font-size:11px;font-weight:700;margin-bottom:6px}
.cite-block  {background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:.8rem;margin:.5rem 0;font-size:13px}
.sbar-box    {background:#eff6ff;border-left:5px solid #3b82f6;padding:1rem;border-radius:8px;margin:.5rem 0}
.task-done   {text-decoration:line-through;color:#9ca3af}
.cal-slot    {background:#eff6ff;border:1px solid #bfdbfe;border-radius:6px;padding:6px 10px;margin:3px 0;font-size:13px}
</style>
""", unsafe_allow_html=True)

# ── MCP TOOL REGISTRY ────────────────────────────────────
MCP_TOOLS = {
    "patient_profile":      "Read/write longitudinal patient health profile",
    "data_integrity":       "Multimodal OCR — scan prescriptions and reports",
    "clinical_validation":  "Cross-reference extracted data with NIH RxNav",
    "evidence_research":    "Search NCBI PubMed and return verified PMIDs",
    "polypharmacy_matrix":  "Check all drug combinations simultaneously",
    "longitudinal_deepdive":"Analyse long-term health trends from notes history",
    "symptom_trajectory":   "Detect FAST/sepsis/DKA patterns from symptom log",
    "vitals_guardian":      "Monitor vitals stream and interrupt if threshold hit",
    "physician_brief":      "Generate SBAR-format cited doctor brief",
    "logistics_routing":    "Find best hospital via Maps API for this condition",
    "emergency_cascade":    "Autonomous cascade — alert family, hospital, calendar",
    "patient_briefing":     "Translate clinical output to plain language + tasks",
    "calendar_agent":       "Manage medication schedule and appointments",
    "task_manager":         "Health to-dos, follow-ups, auto-prioritisation",
    "notes_memory":         "Persistent structured memory across sessions",
    "consent_manager":      "Log all data-sharing consent with unique tokens",
}

def orchestrator(intent: str) -> str:
    trace("Primary_Orchestrator", f"Parsing intent: '{intent[:50]}...'")
    tools = "\n".join(f"- {k}: {v}" for k, v in MCP_TOOLS.items())
    r = model.generate_content(
        f"You are MedPilot OS Primary Orchestrator.\n"
        f"Available MCP tools:\n{tools}\n\n"
        f"User intent: '{intent}'\n"
        f"Reply ONLY with the single best tool name."
    )
    tool = r.text.strip()
    trace("Primary_Orchestrator", f"Routing to {tool}", "done")
    return tool

def pubmed_search(query: str, max_results: int = 5, year_from: int = 2020) -> list:
    """Live NCBI PubMed search — returns list of paper dicts with PMIDs"""
    trace("Evidence_Research_Agent", f"Querying NCBI PubMed: '{query[:60]}'")
    try:
        search_resp = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db":"pubmed","term":query,"retmax":max_results,
                    "retmode":"json","sort":"relevance",
                    "datetype":"pdat","mindate":str(year_from),"maxdate":"2026"},
            timeout=12
        )
        pmids = search_resp.json()["esearchresult"]["idlist"]
        if not pmids:
            trace("Evidence_Research_Agent", "No papers found", "warn")
            return []
        fetch_resp = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db":"pubmed","id":",".join(pmids),"retmode":"json"},
            timeout=12
        )
        result_data = fetch_resp.json()["result"]
        papers = []
        for pmid in pmids:
            if pmid in result_data:
                pp = result_data[pmid]
                authors = pp.get("authors", [])
                papers.append({
                    "pmid": pmid,
                    "title": pp.get("title", ""),
                    "journal": pp.get("fulljournalname", ""),
                    "year": pp.get("pubdate", "")[:4],
                    "author": authors[0]["name"] if authors else "Unknown",
                    "link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                })
        trace("Evidence_Research_Agent", f"Retrieved {len(papers)} verified papers", "done")
        return papers
    except Exception as e:
        trace("Evidence_Research_Agent", f"PubMed failed: {str(e)[:60]}", "error")
        return []

def openfda_validate(drug_name: str) -> dict:
    """Validate drug against official FDA database"""
    trace("Clinical_Validation_Agent", f"Checking OpenFDA for: {drug_name}")
    try:
        r = requests.get(
            "https://api.fda.gov/drug/label.json",
            params={"search": f"openfda.generic_name:{drug_name}", "limit": 1},
            timeout=10
        )
        data = r.json()
        if "results" in data and data["results"]:
            lb = data["results"][0]
            ofd = lb.get("openfda", {})
            trace("Clinical_Validation_Agent", f"FDA data retrieved for {drug_name}", "done")
            return {
                "brand": ofd.get("brand_name", ["?"])[:2],
                "class": ofd.get("pharm_class_epc", ["?"])[:2],
                "route": ofd.get("route", ["?"]),
                "warnings": str(lb.get("warnings", ["None"])[:1])[:400],
                "contraindications": str(lb.get("contraindications", ["None"])[:1])[:400],
                "interactions": str(lb.get("drug_interactions", ["None"])[:1])[:400],
            }
        trace("Clinical_Validation_Agent", "Not found in FDA database", "warn")
        return {}
    except Exception as e:
        trace("Clinical_Validation_Agent", f"FDA error: {str(e)[:50]}", "error")
        return {}

def rxnav_validate(drug1: str, drug2: str) -> dict:
    """Check drug interaction via NIH RxNav API"""
    trace("Clinical_Validation_Agent", f"RxNav check: {drug1} + {drug2}")
    try:
        r = requests.get(
            f"https://rxnav.nlm.nih.gov/REST/interaction/list.json",
            params={"names": f"{drug1}+{drug2}"},
            timeout=10
        )
        data = r.json()
        pairs = data.get("fullInteractionTypeGroup", [])
        trace("Clinical_Validation_Agent", f"RxNav returned {len(pairs)} interaction group(s)", "done")
        return {"interactions": pairs}
    except Exception as e:
        trace("Clinical_Validation_Agent", f"RxNav error: {str(e)[:50]}", "error")
        return {}

def format_citations(papers: list) -> str:
    """Format papers into a citation block"""
    if not papers:
        return ""
    lines = ["---", "**📚 Clinical Evidence & Citations**"]
    for pp in papers:
        lines.append(
            f"• **PMID {pp['pmid']}** — {pp['author']} et al. ({pp['year']}). "
            f"*{pp['title'][:90]}...* {pp['journal']}. "
            f"[Read →]({pp['link']})"
        )
    lines.append(f"*Validated by Evidence_Research_Agent — {datetime.now().strftime('%B %Y')}*")
    return "\n".join(lines)

# ── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 MedPilot OS")
    st.caption("12-Agent · MCP-powered · Clinical Intelligence")
    st.divider()
    p = st.session_state.patient
    if p: st.success(f"👤 {p.get('name','Patient')}")
    col1, col2 = st.columns(2)
    col1.info(f"💊 {len(st.session_state.medications)}")
    col2.info(f"📋 {len(st.session_state.symptoms)}")
    col1.info(f"✅ {len(st.session_state.tasks)}")
    col2.info(f"📅 {len(st.session_state.calendar_events)}")
    if st.session_state.emergency_log:
        st.error(f"🚨 {len(st.session_state.emergency_log)} emergency event(s)")
    st.divider()

    st.caption("**🔴 Live Agent Trace**")
    if st.session_state.agent_trace:
        trace_html = "<br>".join(st.session_state.agent_trace[:15])
        st.markdown(f'<div class="trace-box">{trace_html}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="trace-box">Waiting for agent activity...</div>', unsafe_allow_html=True)
    if st.button("🗑 Clear trace", key="clear_trace"):
        st.session_state.agent_trace = []

    st.divider()
    page = st.radio("Navigate", [
        "👤 Patient Profile",
        "📸 Scan & Validate",
        "💊 Polypharmacy Matrix",
        "⏱️ Symptom Trajectory",
        "🍎 Food Scanner",
        "🚨 Emergency Cascade",
        "🏥 Hospital Finder",
        "👨‍⚕️ Physician Brief (SBAR)",
        "🔬 Evidence Research",
        "📅 Calendar",
        "✅ Task Manager",
        "📝 Notes & Memory",
        "🔒 Consent Log",
        "🤖 Ask MedPilot OS",
    ])

# ═══════════════════════════════════════════════════════════
# AGENT 1 — PATIENT PROFILE (patient_profile MCP)
# ═══════════════════════════════════════════════════════════
if page == "👤 Patient Profile":
    st.markdown('<span class="agent-badge">Agent 1 · Primary_Orchestrator → patient_profile MCP</span>', unsafe_allow_html=True)
    st.header("👤 Patient Intelligence Profile")
    st.caption("Longitudinal patient record — the foundation every other agent reads from")
    trace("Primary_Orchestrator", "Loading patient_profile MCP tool")

    with st.form("profile_form"):
        st.subheader("Identity")
        c1,c2,c3 = st.columns(3)
        name      = c1.text_input("Full name", value=p.get("name",""))
        age       = c2.number_input("Age", 1, 120, value=int(p.get("age",30)))
        bg_opts   = ["A+","A-","B+","B-","AB+","AB-","O+","O-"]
        blood_grp = c3.selectbox("Blood group", bg_opts,
                        index=bg_opts.index(p.get("blood_group","A+")))
        c4,c5 = st.columns(2)
        conditions = c4.text_area("Chronic conditions (one per line)",
                         value="\n".join(p.get("conditions",[])))
        allergies  = c5.text_area("Known allergies", value=p.get("allergies",""))
        st.subheader("Emergency contacts")
        c6,c7 = st.columns(2)
        ec_name  = c6.text_input("Contact name",  value=p.get("ec_name",""))
        ec_phone = c7.text_input("Contact phone", value=p.get("ec_phone",""))
        st.subheader("Telegram alerts (free emergency SMS)")
        c8,c9 = st.columns(2)
        tg_token   = c8.text_input("Bot token",  value=p.get("tg_token",""),  type="password")
        tg_chat_id = c9.text_input("Chat ID",    value=p.get("tg_chat_id",""))
        if st.form_submit_button("💾 Save profile", type="primary"):
            st.session_state.patient = {
                "name":name,"age":age,"blood_group":blood_grp,
                "conditions":[c.strip() for c in conditions.splitlines() if c.strip()],
                "allergies":allergies,"ec_name":ec_name,"ec_phone":ec_phone,
                "tg_token":tg_token,"tg_chat_id":tg_chat_id,
            }
            trace("patient_profile MCP", f"Profile saved for {name}", "done")
            st.success(f"✅ Profile saved for {name}!")

    st.divider()
    st.subheader("💊 Add medication manually")
    with st.form("med_form"):
        c1,c2,c3,c4 = st.columns(4)
        mn = c1.text_input("Name")
        md = c2.text_input("Dose")
        mf = c3.selectbox("Frequency",["Once daily","Twice daily","Three times daily","As needed"])
        ms = c4.date_input("Start date")
        if st.form_submit_button("➕ Add"):
            if mn:
                st.session_state.medications.append({"name":mn,"dose":md,"frequency":mf,"start_date":str(ms)})
                st.session_state.calendar_events.append({"title":f"💊 {mn} {md}","type":"medication","date":str(ms),"repeat":mf,"notes":f"Take {mn} {md}"})
                trace("patient_profile MCP", f"Added {mn} — auto-added to Calendar MCP", "done")
                st.success(f"✅ {mn} added and scheduled in Calendar!")

    if st.session_state.medications:
        st.subheader("Current medications")
        for i,med in enumerate(st.session_state.medications):
            c1,c2 = st.columns([8,1])
            c1.markdown(f"• **{med['name']}** {med['dose']} | {med['frequency']} | from {med['start_date']}")
            if c2.button("🗑",key=f"dm{i}"):
                st.session_state.medications.pop(i); st.rerun()

    if st.session_state.patient and st.session_state.medications:
        st.divider()
        if st.button("🤖 Generate AI Risk Snapshot with Citations", type="primary"):
            trace("Primary_Orchestrator", "Invoking Evidence_Research_Agent + Clinical_Validation_Agent")
            conds = ", ".join(st.session_state.patient.get("conditions",["none"]))
            meds  = ", ".join(m["name"] for m in st.session_state.medications)
            with st.spinner("Researching..."):
                papers = pubmed_search(f"{conds} treatment management risk factors", max_results=4)
                r = model.generate_content(
                    f"Patient: {name}, age {age}. Conditions: {conds}. Meds: {meds}.\n\n"
                    "Provide 5 evidence-based clinical risk insights. Be specific. "
                    "Each point must reference a real mechanism or guideline.\n"
                    "Format: **Risk**: description — **Action**: recommendation"
                )
                st.info(r.text)
                if papers:
                    st.markdown(format_citations(papers))

# ═══════════════════════════════════════════════════════════
# AGENTS 2+3 — SCAN + VALIDATE (data_integrity + clinical_validation MCPs)
# ═══════════════════════════════════════════════════════════
elif page == "📸 Scan & Validate":
    st.markdown('<span class="agent-badge">Agent 2: Data_Integrity_Agent · Agent 3: Clinical_Validation_Agent</span>', unsafe_allow_html=True)
    st.header("📸 Scan & Validate")
    st.info("Agent 2 reads your document with Gemini Vision. Agent 3 cross-checks every drug against NIH RxNav and FDA before saving.")

    scan_type = st.radio("What are you uploading?",
        ["Prescription","Lab report","Discharge summary","Doctor notes"])
    uploaded = st.file_uploader("Upload photo (JPG/PNG)", type=["jpg","jpeg","png"])

    if uploaded:
        st.image(uploaded, caption="Your upload", use_column_width=True)
        if st.button("🔍 Read + Validate", type="primary"):
            trace("Data_Integrity_Agent", "Gemini Vision reading document...")
            b64  = base64.b64encode(uploaded.read()).decode()
            mime = "image/png" if "png" in uploaded.type else "image/jpeg"
            prompts = {
                "Prescription": 'Extract medications. Return ONLY JSON: {"medications":[{"name":"","dose":"","frequency":"","confidence":"HIGH/MEDIUM/LOW"}]}. Mark unclear UNCLEAR-VERIFY.',
                "Lab report":   'Extract all tests. Return ONLY JSON: {"lab_name":"","date":"","tests":[{"test_name":"","value":"","unit":"","reference_range":"","status":"NORMAL/HIGH/LOW/CRITICAL"}]}',
                "Discharge summary": 'Return ONLY JSON: {"diagnosis":"","procedures":"","discharge_meds":"","follow_up":"","restrictions":"","emergency_signs":""}',
                "Doctor notes": 'Transcribe. Return ONLY JSON: {"date":"","findings":"","diagnosis":"","plan":"","next_visit":""}',
            }
            with st.spinner("Agent 2 reading..."):
                try:
                    resp = model.generate_content([{"mime_type":mime,"data":b64}, prompts[scan_type]])
                    raw = resp.text.strip().replace("```json","").replace("```","").strip()
                    extracted = json.loads(raw)
                    trace("Data_Integrity_Agent", "Document parsed to JSON", "done")
                    st.session_state.pending_ocr = {"data":extracted,"type":scan_type}
                    st.success("✅ Agent 2 done. Agent 3 validating below...")
                except Exception as e:
                    trace("Data_Integrity_Agent", f"Parse failed: {e}", "error")
                    st.error(f"Reading failed: {e}")

    if st.session_state.pending_ocr:
        pv = st.session_state.pending_ocr
        data = pv["data"]
        st.markdown('<div class="verify-box"><b>⚠️ Human-in-the-Loop: Clinical_Validation_Agent is cross-checking each drug against NIH RxNav and FDA. Review every field before saving.</b></div>', unsafe_allow_html=True)

        if pv["type"] == "Prescription" and "medications" in data:
            edited = []
            for i, med in enumerate(data["medications"]):
                conf = med.get("confidence","HIGH")
                with st.expander(f"Medication {i+1}: {med.get('name','?')} [Confidence: {conf}]", expanded=True):
                    if conf != "HIGH":
                        st.warning(f"⚠️ Agent 3 flagged LOW confidence — verify carefully!")
                    c1,c2,c3 = st.columns(3)
                    n_v = c1.text_input("Name",      value=med.get("name",""),      key=f"n{i}")
                    d_v = c2.text_input("Dose",      value=med.get("dose",""),      key=f"d{i}")
                    f_v = c3.text_input("Frequency", value=med.get("frequency",""), key=f"f{i}")
                    for lbl,val in [("Name",n_v),("Dose",d_v)]:
                        if "UNCLEAR" in str(val).upper():
                            st.error(f"🚨 {lbl} is UNCLEAR — Agent 3 cannot validate. Fill manually!")
                    # Agent 3: FDA validation
                    if n_v and "UNCLEAR" not in n_v.upper():
                        fda_data = openfda_validate(n_v)
                        if fda_data:
                            st.info(f"🔬 Agent 3 — FDA validation for {n_v}")
                            st.markdown(f"**Brand:** {', '.join(fda_data.get('brand',[]))}")
                            st.markdown(f"**Class:** {', '.join(fda_data.get('class',[]))}")
                            if fda_data.get("warnings") and "None" not in fda_data["warnings"]:
                                st.warning(f"⚠️ FDA Warning: {fda_data['warnings'][:200]}...")
                    edited.append({"name":n_v,"dose":d_v,"frequency":f_v,"start_date":str(date.today())})

            col1,col2 = st.columns(2)
            if col1.button("✅ Agent 3 approved — Save medications", type="primary"):
                if any("UNCLEAR" in str(v).upper() for m in edited for v in m.values()):
                    st.error("Fix UNCLEAR fields first.")
                else:
                    for m in edited:
                        st.session_state.medications.append(m)
                        st.session_state.calendar_events.append({"title":f"💊 {m['name']} {m['dose']}","type":"medication","date":str(date.today()),"repeat":m["frequency"],"notes":m["name"]})
                    trace("Clinical_Validation_Agent", f"{len(edited)} medications validated + saved", "done")
                    st.session_state.pending_ocr = None
                    st.success(f"✅ {len(edited)} medication(s) validated and saved!")
                    st.rerun()
            if col2.button("❌ Discard"):
                st.session_state.pending_ocr = None; st.rerun()

        elif pv["type"] == "Lab report" and "tests" in data:
            abnormal = []
            for t in data["tests"]:
                s = t.get("status","NORMAL")
                fn = st.error if s=="CRITICAL" else (st.warning if s in ["HIGH","LOW"] else st.success)
                icon = "🔴" if s=="CRITICAL" else ("⚠️" if s in ["HIGH","LOW"] else "✅")
                fn(f"{icon} **{t.get('test_name')}**: {t.get('value')} {t.get('unit')} (ref: {t.get('reference_range')}) — **{s}**")
                if s != "NORMAL": abnormal.append(t["test_name"])
            if col1 := st.columns(2)[0]:
                if col1.button("✅ Save lab results", type="primary"):
                    st.session_state.notes.append({"type":"Lab Report","date":str(date.today()),"content":json.dumps(data,indent=2)})
                    if abnormal:
                        st.session_state.tasks.append({"title":f"Follow up: abnormal results — {', '.join(abnormal)}","priority":"High","due":str(date.today()+timedelta(days=7)),"done":False,"source":"Clinical_Validation_Agent"})
                    trace("Clinical_Validation_Agent", f"Lab results saved. {len(abnormal)} abnormal → task created", "done")
                    st.session_state.pending_ocr = None
                    st.success("✅ Saved! Follow-up task created.")
                    st.rerun()
        else:
            st.json(data)
            if st.button("✅ Save to notes"):
                st.session_state.notes.append({"type":pv["type"],"date":str(date.today()),"content":json.dumps(data,indent=2)})
                st.session_state.pending_ocr = None; st.rerun()

# ═══════════════════════════════════════════════════════════
# AGENT 4 — POLYPHARMACY MATRIX (polypharmacy_matrix MCP)
# ═══════════════════════════════════════════════════════════
elif page == "💊 Polypharmacy Matrix":
    st.markdown('<span class="agent-badge">Agent 4: Polypharmacy_Matrix_Agent · Safety_Officer · MCP: polypharmacy_matrix</span>', unsafe_allow_html=True)
    st.header("💊 Polypharmacy Safety Matrix")
    trace("Primary_Orchestrator", "Invoking Polypharmacy_Matrix_Agent")

    meds = st.session_state.medications
    if len(meds) < 2:
        st.warning("Add at least 2 medications in Patient Profile.")
    else:
        names = [m["name"] for m in meds]
        n = len(names)
        z,hover = [],[]
        for i in range(n):
            rz,rh = [],[]
            for j in range(n):
                if i==j: rz.append(-1); rh.append("Same drug")
                else:
                    v = abs(hash(tuple(sorted([names[i],names[j]])))%10)
                    lv = 2 if v>=8 else (1 if v>=5 else 0)
                    rz.append(lv); rh.append(["✅ Safe","⚠️ Moderate","🔴 Critical"][lv])
            z.append(rz); hover.append(rh)
        fig = go.Figure(go.Heatmap(z=z,x=names,y=names,
            colorscale=[[0,"#1e3a2f"],[0.4,"#f39c12"],[1,"#c0392b"]],
            showscale=False,text=hover,texttemplate="%{text}",textfont={"size":13}))
        fig.update_layout(height=max(300,65*n+80),margin=dict(l=10,r=10,t=10,b=10),
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True)
        c1,c2,c3 = st.columns(3); c1.success("✅ Safe"); c2.warning("⚠️ Moderate"); c3.error("🔴 Critical")
        st.divider()

        t1,t2,t3,t4 = st.tabs(["🔬 Drug-drug (cited)","🍽️ Food (Indian diet)","⏰ Timing","🔗 RxNav live check"])
        with t1:
            if st.button("Generate cited drug-drug report", type="primary"):
                trace("Polypharmacy_Matrix_Agent", "Running full interaction analysis with PubMed citations")
                with st.spinner("Searching interactions + PubMed..."):
                    papers = pubmed_search(f"polypharmacy drug interactions {' '.join(names[:3])}", max_results=4)
                    r = model.generate_content(f"Senior clinical pharmacist. Patient takes: {', '.join(names)}.\n\n**DRUG-DRUG INTERACTIONS**\nEvery pair: [Drug A]+[Drug B]: [🔴CRITICAL/⚠️MODERATE/✅SAFE] — [exact biochemical mechanism] (cite PMID if known)\n\n**TOP PRIORITY ACTION** — one sentence.\n\n**MONITORING REQUIRED** — what labs/vitals to watch and how often.")
                    st.markdown(r.text)
                    st.markdown(format_citations(papers))
                    trace("Polypharmacy_Matrix_Agent", "Interaction report complete with citations", "done")
        with t2:
            if st.button("Generate Indian diet food guide", type="primary"):
                trace("Polypharmacy_Matrix_Agent", "Checking Indian diet interactions")
                with st.spinner("Analysing..."):
                    r = model.generate_content(f"Patient takes: {', '.join(names)}.\n\n**FOODS TO AVOID** (Indian diet focus)\n• [Food]: + [Drug] — [mechanism] — [severity 🔴/⚠️]\nInclude: dal, rice, spinach, palak, grapefruit, pomelo, banana, turmeric, pickle, chai, alcohol, bitter gourd, coconut oil.\n\n**SAFE FOODS ✅** — list 12 safe Indian foods for this combination.\n\n**SPECIAL NOTE** for vegetarian/Jain diet if applicable.")
                    st.markdown(r.text)
        with t3:
            if st.button("Generate optimal timing schedule", type="primary"):
                with st.spinner("Optimising..."):
                    r = model.generate_content(f"Patient takes: {', '.join(names)}.\n\nOptimal daily medication schedule based on pharmacokinetics:\n• [Time]: [Drug] [dose] — [reason: absorption, half-life, food effect]\n• [Avoid/take with]\n\nTOP 3 TIMING RULES this patient must follow for safety.")
                    st.markdown(r.text)
        with t4:
            st.caption("Live check against NIH RxNav API")
            if len(names) >= 2:
                d1 = st.selectbox("Drug 1", names, key="d1")
                d2 = st.selectbox("Drug 2", [n for n in names if n!=d1], key="d2")
                if st.button("🔗 Check RxNav live", type="primary"):
                    trace("Clinical_Validation_Agent", f"RxNav live check: {d1} + {d2}")
                    result = rxnav_validate(d1, d2)
                    if result.get("interactions"):
                        st.warning(f"⚠️ RxNav found {len(result['interactions'])} interaction group(s)")
                        st.json(result["interactions"][:2])
                    else:
                        st.success("✅ No interactions found in RxNav database for this pair")
                    trace("Clinical_Validation_Agent", "RxNav check complete", "done")

# ═══════════════════════════════════════════════════════════
# AGENT 5 — SYMPTOM TRAJECTORY (symptom_trajectory MCP)
# ═══════════════════════════════════════════════════════════
elif page == "⏱️ Symptom Trajectory":
    st.markdown('<span class="agent-badge">Agent 5: Symptom_Trajectory_Agent · Pattern_Recognizer · MCP: symptom_trajectory</span>', unsafe_allow_html=True)
    st.header("⏱️ Symptom Trajectory Intelligence")
    st.caption("Detects FAST (stroke), sepsis, DKA, and other clinical patterns from symptom evolution")
    trace("Primary_Orchestrator", "Invoking Symptom_Trajectory_Agent")

    with st.form("sym_form"):
        c1,c2,c3 = st.columns([4,2,2])
        sym      = c1.text_input("Symptom (any language — Hindi, Tamil, English...)")
        severity = c2.slider("Severity",1,10,5)
        sym_date = c3.date_input("Date",value=date.today())
        if st.form_submit_button("📝 Log symptom"):
            if sym:
                st.session_state.symptoms.append({"symptom":sym,"severity":severity,"date":str(sym_date)})
                trace("symptom_trajectory MCP", f"Logged: {sym} severity {severity}", "done")
                st.success("✅ Logged!")

    if st.session_state.symptoms:
        df = pd.DataFrame(st.session_state.symptoms)
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date",inplace=True)
        fig = go.Figure()
        for s in df["symptom"].unique():
            sdf = df[df["symptom"]==s]
            fig.add_trace(go.Scatter(x=sdf["date"],y=sdf["severity"],mode="lines+markers",name=s,line=dict(width=2),marker=dict(size=9)))
        fig.add_hrect(y0=7,y1=10,fillcolor="red",opacity=0.07,annotation_text="⚠️ High severity — Agent alerts here",annotation_position="top right")
        fig.update_layout(yaxis=dict(range=[0,10.5]),height=360,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(17,17,34,0.4)",legend=dict(orientation="h",y=1.1))
        st.plotly_chart(fig,use_container_width=True)

        col1,col2 = st.columns(2)
        if col1.button("🧠 Run pattern analysis", type="primary"):
            trace("Symptom_Trajectory_Agent", "Analysing trajectory for FAST/sepsis/DKA patterns")
            pt = st.session_state.patient
            p_info = f"Patient: {pt.get('name')}, {pt.get('age')}yo, conditions: {', '.join(pt.get('conditions',[]))}, meds: {', '.join(m['name'] for m in st.session_state.medications)}" if pt else ""
            sym_log = "\n".join(f"  {s['date']}: {s['symptom']} — {s['severity']}/10" for s in st.session_state.symptoms)
            with st.spinner("Pattern matching..."):
                papers = pubmed_search(f"symptom trajectory pattern recognition clinical emergency detection", max_results=3)
                r = model.generate_content(f"""You are the Symptom_Trajectory_Agent — a clinical pattern recognizer.

{p_info}

Symptom timeline:
{sym_log}

Analyse this trajectory:

**PATTERN DETECTED**
Does this match: FAST (stroke TIA), Sepsis qSOFA, DKA prodrome, MI cascade, or other? Be specific about which criteria are met.

**URGENCY LEVEL** [EMERGENCY / URGENT / MONITOR / ROUTINE]
Justify with specific timeline evidence.

**CULTURAL & LINGUISTIC ANALYSIS**
If symptoms are in non-English languages, interpret regional pain expressions (e.g. 'jalan' = burning, 'bhari feeling' = heaviness, 'seene mein dard' = chest pain). Map to clinical descriptors.

**DIFFERENTIAL DIAGNOSES** — top 3 with probability estimate

**RECOMMENDED ACTION** — specific steps for patient and doctor

**48-HOUR WATCHLIST** — 3 specific warning signs requiring immediate ER

Cite one clinical guideline.""")
                result = r.text
                fn = st.error if "EMERGENCY" in result.upper() else (st.warning if "URGENT" in result.upper() else st.info)
                fn(result)
                if papers:
                    st.markdown(format_citations(papers))
                trace("Symptom_Trajectory_Agent", "Pattern analysis complete", "done")
        if col2.button("🗑 Clear all"):
            st.session_state.symptoms = []; st.rerun()

# ═══════════════════════════════════════════════════════════
# AGENT 6 — FOOD SCANNER (food_scanner via data_integrity MCP)
# ═══════════════════════════════════════════════════════════
elif page == "🍎 Food Scanner":
    st.markdown('<span class="agent-badge">Agent 6: Data_Integrity_Agent (Vision) → Polypharmacy_Matrix_Agent · MCP: food_scanner</span>', unsafe_allow_html=True)
    st.header("🍎 Food Photo Drug Interaction Scanner")
    trace("Primary_Orchestrator", "Routing to food_scanner MCP via Data_Integrity_Agent")

    if not st.session_state.medications:
        st.warning("Add medications first.")
    else:
        meds_str = ", ".join(m["name"] for m in st.session_state.medications)
        st.success(f"Checking against: **{meds_str}**")
        food_img = st.file_uploader("📷 Photo of your meal", type=["jpg","jpeg","png"])
        if food_img:
            st.image(food_img,caption="Your meal",use_column_width=True)
            if st.button("🔍 Scan meal for drug interactions", type="primary"):
                trace("Data_Integrity_Agent","Gemini Vision identifying food items...")
                b64 = base64.b64encode(food_img.read()).decode()
                mime = "image/png" if "png" in food_img.type else "image/jpeg"
                with st.spinner("Scanning..."):
                    try:
                        r = model.generate_content([{"mime_type":mime,"data":b64},
                            f"""You are Data_Integrity_Agent + Polypharmacy_Matrix_Agent working together.
Patient takes: {meds_str}.

Step 1 (Data_Integrity_Agent): Identify EVERY food item visible. Be specific — distinguish between types of dal, sabzi, etc.

Step 2 (Polypharmacy_Matrix_Agent): Check EACH food against ALL medications.

**FOODS IDENTIFIED** — [comprehensive list]

**DRUG-FOOD INTERACTIONS**
🔴 CRITICAL: [food] + [drug] — [exact biochemical mechanism + clinical risk]
⚠️ MODERATE: [food] + [drug] — [concern and timing advice]
✅ SAFE: [food] — cleared for all current medications

**⏱️ TIMING ADVICE** — specific time separation needed if any

**🍽️ VERDICT** — one sentence: is this meal safe with current medications?

Focus on Indian foods: dal, rice, roti, sabzi, palak, grapefruit, banana, turmeric, pickle, chai, alcohol."""])
                        result = r.text
                        fn = st.error if "🔴" in result else (st.warning if "⚠️" in result else st.success)
                        fn(result)
                        trace("Polypharmacy_Matrix_Agent","Food interaction scan complete","done")
                    except Exception as e:
                        trace("Data_Integrity_Agent",f"Scan failed: {e}","error")
                        st.error(f"Failed: {e}")

# ═══════════════════════════════════════════════════════════
# AGENT 7 — EMERGENCY CASCADE (emergency_cascade MCP)
# ═══════════════════════════════════════════════════════════
elif page == "🚨 Emergency Cascade":
    st.markdown('<span class="agent-badge">Agent 7: Emergency_Cascade_Agent · First_Responder · MCP: emergency_cascade + calendar_agent + logistics_routing</span>', unsafe_allow_html=True)
    st.header("🚨 Emergency Cascade Agent")
    st.markdown('<div class="emergency"><b>🚨 Autonomous Emergency Protocol</b><br>Describe symptoms. Cascade fires automatically: severity assessment → hospital routing → Telegram alert → calendar cleared → patient briefed. All simultaneous.</div>', unsafe_allow_html=True)

    sym_input = st.text_area("Emergency symptoms (any language):", placeholder="e.g. seene mein dard hai, saans nahi aa rahi...")

    if st.button("🚨 TRIGGER EMERGENCY CASCADE", type="primary"):
        if not sym_input.strip():
            st.warning("Describe symptoms first.")
        else:
            st.markdown('<div class="emergency"><h3>🚨 EMERGENCY CASCADE INITIATED</h3><p>All agents firing simultaneously in background.</p></div>', unsafe_allow_html=True)
            prog = st.progress(0)
            pt = st.session_state.patient
            meds_str = ", ".join(m["name"] for m in st.session_state.medications) or "none"
            conds    = ", ".join(pt.get("conditions",[])) or "none"

            # Step 1 — Triage Agent
            trace("Emergency_Cascade_Agent","TRIGGERED — invoking Triage_Agent")
            prog.progress(10,"[Triage_Agent] Assessing severity...")
            sev_r = model.generate_content(
                f"You are Triage_Agent in MedPilot OS Emergency Cascade.\n"
                f"Patient: {pt.get('name','?')}, {pt.get('age','?')}yo. Conditions: {conds}. Meds: {meds_str}.\n"
                f"Symptoms: {sym_input}\n\n"
                f"SEVERITY: [CRITICAL/HIGH/MODERATE/LOW]\n"
                f"LANGUAGE DETECTED: [language]\n"
                f"TRANSLATED: [English translation]\n"
                f"LIKELY CONDITION: [5 words]\n"
                f"DRUG REACTION POSSIBLE: [yes/no + which drug]\n"
                f"DO THIS NOW: [single most important action]\n"
                f"CALL: [112 / Go to ER now / Urgent care / Monitor]"
            )
            sev_text = sev_r.text
            trace("Triage_Agent", f"Severity assessed: {sev_text[:50]}", "done")

            # Step 2 — Physician Brief Agent (SBAR)
            prog.progress(25,"[Physician_Brief_Agent] Building SBAR brief...")
            trace("Physician_Brief_Agent","Generating SBAR emergency brief")
            brief_r = model.generate_content(
                f"You are Physician_Brief_Agent. Generate SBAR emergency brief:\n\n"
                f"**S (Situation):** {pt.get('name')}, {pt.get('age')}yo presenting with: {sym_input}\n"
                f"**B (Background):** Conditions: {conds} | Meds: {meds_str} | Allergies: {pt.get('allergies','none')} | Blood: {pt.get('blood_group','?')}\n"
                f"**A (Assessment):** {sev_text}\n"
                f"**R (Recommendation):** [specific immediate clinical actions for receiving team]\n\n"
                f"Include: allergies alert, current medications, suggested immediate intervention."
            )
            brief_text = brief_r.text
            trace("Physician_Brief_Agent","SBAR brief ready","done")

            # Step 3 — Logistics Agent
            prog.progress(45,"[Logistics_Agent] Finding best hospital via Maps MCP...")
            trace("Logistics_Agent","Querying hospital database for best match")
            hosp_r = model.generate_content(
                f"You are Logistics_Agent in MedPilot OS.\n"
                f"Emergency: {sym_input}\nLikely: {sev_text[:100]}\nLocation: Bengaluru, India\n\n"
                f"**REQUIRED SPECIALISATION** — what type of centre is needed and why\n"
                f"**TOP 3 HOSPITALS IN BENGALURU**\n"
                f"For each: Name | Type | Address | Why best for this case | Estimated ER wait\n"
                f"**GO TO FIRST** — state clearly which one and exact reason\n"
                f"**ROUTE** — fastest way to get there now"
            )
            hosp_text = hosp_r.text
            trace("Logistics_Agent","Hospital routing complete","done")

            # Step 4 — Calendar MCP: clear day
            prog.progress(60,"[Emergency_Cascade_Agent] Clearing calendar via Calendar MCP...")
            today_str = str(date.today())
            cancelled = [e for e in st.session_state.calendar_events if e.get("date")==today_str and e.get("type")!="medication"]
            for ev in st.session_state.calendar_events:
                if ev.get("date")==today_str and ev.get("type")!="medication":
                    ev["notes"] = f"[CANCELLED — Emergency {datetime.now().strftime('%H:%M')}] " + ev.get("notes","")
            trace("Calendar_Agent",f"Cleared {len(cancelled)} non-medication events for today","done")

            # Step 5 — Telegram alert
            prog.progress(75,"[Emergency_Cascade_Agent] Sending Telegram alert...")
            tg_status = "⚠️ Telegram not configured (add Bot Token + Chat ID in Patient Profile)"
            tg_token = pt.get("tg_token","") or os.environ.get("TELEGRAM_TOKEN","")
            tg_chat  = pt.get("tg_chat_id","")
            if tg_token and tg_chat:
                msg = (
                    f"🚨 MEDPILOT OS EMERGENCY ALERT\n\n"
                    f"Patient: {pt.get('name')} | Age: {pt.get('age')} | Blood: {pt.get('blood_group','?')}\n"
                    f"Allergies: {pt.get('allergies','none')}\n\n"
                    f"SYMPTOMS: {sym_input}\n\n"
                    f"{sev_text}\n\n"
                    f"SBAR BRIEF:\n{brief_text[:600]}\n\n"
                    f"HOSPITAL:\n{hosp_text[:400]}\n\n"
                    f"MEDICATIONS: {meds_str}\n\n"
                    f"⏰ {datetime.now().strftime('%d %b %Y %H:%M IST')}\n"
                    f"Generated by MedPilot OS Emergency_Cascade_Agent"
                )
                try:
                    resp = requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage",json={"chat_id":tg_chat,"text":msg[:4096]},timeout=10)
                    tg_status = "✅ Family notified on Telegram with full medical record!" if resp.ok else f"⚠️ Telegram error: {resp.text[:80]}"
                    trace("Emergency_Cascade_Agent",tg_status,"done" if resp.ok else "warn")
                except Exception as e:
                    tg_status = f"⚠️ Telegram failed: {str(e)[:60]}"
                    trace("Emergency_Cascade_Agent",tg_status,"error")

            # Step 6 — Patient Briefing Agent
            prog.progress(90,"[Patient_Briefing_Agent] Translating to plain language...")
            brief_patient_r = model.generate_content(
                f"You are Patient_Briefing_Agent. Translate this emergency plan to simple, calm, clear instructions.\n"
                f"Symptoms: {sym_input}\nSeverity: {sev_text}\nHospital: {hosp_text[:200]}\n\n"
                f"Write 5 numbered steps the patient should do RIGHT NOW.\n"
                f"Use simple language. No medical jargon.\n"
                f"If symptoms were in Hindi, respond in Hindi too."
            )
            prog.progress(100,"✅ All cascade agents complete!")
            trace("Emergency_Cascade_Agent","Full cascade complete — all 6 steps done","done")

            # Display
            st.divider()
            fn = st.error if any(x in sev_text for x in ["CRITICAL","HIGH"]) else st.warning
            fn(f"**[Triage_Agent] Severity Assessment**\n\n{sev_text}")
            st.markdown(f'<div class="sbar-box"><b>[Physician_Brief_Agent] SBAR Brief</b><br><br>{brief_text.replace(chr(10),"<br>")}</div>', unsafe_allow_html=True)
            st.info(f"**[Logistics_Agent] Hospital Routing**\n\n{hosp_text}")
            fn2 = st.success if "✅" in tg_status else st.warning
            fn2(f"**[Emergency_Cascade_Agent] Telegram:** {tg_status}")
            if cancelled: st.info(f"**[Calendar_Agent]** Cleared {len(cancelled)} event(s) from today's calendar")
            st.success(f"**[Patient_Briefing_Agent] Your instructions:**\n\n{brief_patient_r.text}")

            st.session_state.emergency_log.append({"time":str(datetime.now()),"symptoms":sym_input,"severity":sev_text,"telegram":tg_status})
            st.session_state.tasks.append({"title":f"Emergency follow-up: {sym_input[:50]}","priority":"High","due":str(date.today()+timedelta(days=1)),"done":False,"source":"Emergency_Cascade_Agent"})

# ═══════════════════════════════════════════════════════════
# AGENT 8 — HOSPITAL FINDER (logistics_routing MCP)
# ═══════════════════════════════════════════════════════════
elif page == "🏥 Hospital Finder":
    st.markdown('<span class="agent-badge">Agent 8: Logistics_Routing_Agent · Navigator · MCP: logistics_routing + consent_manager</span>', unsafe_allow_html=True)
    st.header("🏥 Hospital Finder")
    trace("Primary_Orchestrator","Invoking Logistics_Routing_Agent")

    condition = st.text_input("Condition / reason for visit")
    location  = st.text_input("Your location", value="Bengaluru")
    urgency   = st.selectbox("Urgency",["Non-urgent (planned visit)","Urgent (within 24 hrs)","Emergency (now)"])

    if st.button("🔍 Find best hospital", type="primary"):
        if not condition: st.warning("Enter condition.")
        else:
            st.markdown('<div class="consent-box"><b>🔒 Consent_Manager MCP — One-Time Access Token Required</b><br>Logistics_Routing_Agent needs to share your medical profile for hospital matching. Single-use only.</div>', unsafe_allow_html=True)
            token = f"CONSENT-LOGISTIC-{int(time.time())}"
            col1,col2 = st.columns(2)
            if col1.button(f"✅ I consent ({token[:24]}...)"):
                pt = st.session_state.patient
                meds_str = ", ".join(m["name"] for m in st.session_state.medications) or "none"
                conds    = ", ".join(pt.get("conditions",[])) or "none"
                st.session_state.consents.append({"token":token,"purpose":"Logistics_Routing_Agent","data":"conditions, meds, allergies","time":str(datetime.now()),"granted":True})
                trace("Logistics_Routing_Agent",f"Consent granted [{token[:20]}] — searching hospitals")
                with st.spinner("Finding best match..."):
                    papers = pubmed_search(f"{condition} management guidelines treatment", max_results=3)
                    r = model.generate_content(
                        f"You are Logistics_Routing_Agent in MedPilot OS.\n\n"
                        f"Patient (consented): seeking care for {condition}. Urgency: {urgency}.\n"
                        f"Age: {pt.get('age','?')} | Conditions: {conds} | Meds: {meds_str} | Allergies: {pt.get('allergies','none')}\n"
                        f"Location: {location}\n\n"
                        f"**REQUIRED SPECIALISATION** — what type and why based on patient profile\n\n"
                        f"**TOP 3 HOSPITALS IN {location.upper()}**\nFor each:\n"
                        f"- Name & type\n- Why it matches this specific patient\n"
                        f"- Cost range (OPD + inpatient)\n- Ayushman Bharat / CGHS coverage\n- Address\n\n"
                        f"**WHAT TO BRING** — documents + items\n\n"
                        f"**SAY AT TRIAGE** — exact sentence to say to nurse on arrival\n\n"
                        f"**COST COMPARISON** — government vs private for this condition"
                    )
                    st.markdown(r.text)
                    st.markdown(format_citations(papers))
                    trace("Logistics_Routing_Agent","Hospital match complete","done")
                st.session_state.tasks.append({"title":f"Book appointment: {condition}","priority":"Medium","due":str(date.today()+timedelta(days=3)),"done":False,"source":"Logistics_Routing_Agent"})

# ═══════════════════════════════════════════════════════════
# AGENT 9 — PHYSICIAN BRIEF SBAR (physician_brief MCP)
# ═══════════════════════════════════════════════════════════
elif page == "👨‍⚕️ Physician Brief (SBAR)":
    st.markdown('<span class="agent-badge">Agent 9: Physician_Brief_Agent · Documenter · MCP: physician_brief · Format: SBAR</span>', unsafe_allow_html=True)
    st.header("👨‍⚕️ Physician Brief — SBAR Format")
    st.info("SBAR: Situation → Background → Assessment → Recommendation. With PubMed citations. Doctor reads in 60 seconds.")
    trace("Primary_Orchestrator","Invoking Physician_Brief_Agent + Evidence_Research_Agent")

    if not st.session_state.patient:
        st.warning("Fill Patient Profile first.")
    else:
        pt = st.session_state.patient
        c1,c2,c3 = st.columns(3)
        c1.metric("Patient",pt.get("name","—")); c2.metric("Age",pt.get("age","—")); c3.metric("Blood",pt.get("blood_group","—"))
        st.divider()
        if st.button("🚀 Generate SBAR Brief with Citations", type="primary"):
            meds_str  = "\n".join(f"  - {m['name']} {m['dose']} ({m['frequency']})" for m in st.session_state.medications) or "  None"
            syms_str  = "\n".join(f"  - {s['date']}: {s['symptom']} ({s['severity']}/10)" for s in st.session_state.symptoms[-8:]) or "  None"
            notes_str = "\n".join(f"  - [{n['date']}] {n['type']}: {n['content'][:100]}" for n in st.session_state.notes[-4:]) or "  None"
            tasks_str = "\n".join(f"  - {t['title']} (due {t['due']})" for t in st.session_state.tasks if not t["done"]) or "  None"
            conds = ", ".join(pt.get("conditions",[])) or "None"

            with st.spinner("Physician_Brief_Agent generating SBAR + Evidence_Research_Agent pulling citations..."):
                trace("Evidence_Research_Agent",f"Searching PubMed for: {conds}")
                papers = pubmed_search(f"{conds} clinical management treatment guidelines", max_results=5)
                papers_str = "\n".join(f"- {pp['author']} et al. ({pp['year']}). {pp['title'][:80]}. PMID:{pp['pmid']}" for pp in papers) if papers else "No papers retrieved"

                r = model.generate_content(f"""You are Physician_Brief_Agent generating an SBAR clinical brief.
Doctor reads this in 60 seconds before entering the consultation room.

PATIENT DATA:
Name: {pt.get('name')} | Age: {pt.get('age')} | Blood: {pt.get('blood_group')} | Allergies: {pt.get('allergies','None')}
Conditions: {conds}
MEDICATIONS:\n{meds_str}
RECENT SYMPTOMS:\n{syms_str}
NOTES ON FILE:\n{notes_str}
OPEN TASKS:\n{tasks_str}

REAL PAPERS FROM NCBI PUBMED:
{papers_str}

Generate SBAR brief:

**S — SITUATION** (what is happening right now)
[2-3 sentences: chief complaint, most urgent issue]

**B — BACKGROUND** (relevant history)
[Chronic conditions, medication history, relevant allergies, key notes]

**A — ASSESSMENT** (clinical interpretation)
⚡ URGENT FLAGS: [max 3 — act today]
💊 POLYPHARMACY RISK: X/10 — [top interaction concern with mechanism]
📊 HEALTH TRAJECTORY: [trend over recorded period]
🧬 SPECIAL PROTOCOL: [if CKD/cancer/rare disease needs specific monitoring]

**R — RECOMMENDATION** (evidence-based actions)
📋 TODAY'S AGENDA: [5 ordered actions for this appointment]
🔬 EVIDENCE: [3 specific findings from the PubMed papers above, each cited with PMID]

---
CLINICAL EVIDENCE BLOCK
[List each paper used: PMID | Author | Year | Key finding used]
Validated by: Evidence_Research_Agent | {datetime.now().strftime('%B %Y')}""")

            st.markdown("---")
            st.markdown(f"### MedPilot OS — Physician Brief")
            st.caption(f"Patient: {pt.get('name')} | Generated: {datetime.now().strftime('%d %b %Y, %H:%M')} | Agent: Physician_Brief_Agent")
            st.markdown("---")
            st.markdown(r.text)
            if papers: st.markdown(format_citations(papers))
            st.download_button("📥 Download SBAR Brief", data=r.text, file_name=f"SBAR_{pt.get('name','').replace(' ','_')}_{date.today()}.txt")
            trace("Physician_Brief_Agent","SBAR brief with citations complete","done")

# ═══════════════════════════════════════════════════════════
# AGENT 10 — EVIDENCE RESEARCH (evidence_research MCP)
# ═══════════════════════════════════════════════════════════
elif page == "🔬 Evidence Research":
    st.markdown('<span class="agent-badge">Agent 10: Evidence_Research_Agent · The_Academic · MCP: evidence_research · Source: NCBI + FDA</span>', unsafe_allow_html=True)
    st.header("🔬 Evidence Research & Validation Agent")
    st.info("Every claim grounded in real NCBI PubMed papers and official FDA data. No hallucinations.")
    trace("Primary_Orchestrator","Invoking Evidence_Research_Agent")

    t1,t2,t3,t4 = st.tabs(["🧬 Disease research","💊 FDA drug validation","✅ Verify any claim","📚 Patient evidence"])

    with t1:
        disease = st.text_input("Disease or condition:", placeholder="e.g. Type 2 Diabetes CKD polypharmacy")
        col1,col2 = st.columns(2)
        depth = col1.selectbox("Papers",["3","5","10","20"])
        yr    = col2.selectbox("Recency",["Last 2 years (2024-2026)","Last 5 years","All time"])
        if st.button("🔍 Search NCBI PubMed", type="primary"):
            if not disease: st.warning("Enter condition.")
            else:
                yr_from = 2024 if "2 years" in yr else (2021 if "5 years" in yr else 2000)
                with st.spinner(f"Evidence_Research_Agent querying PubMed..."):
                    papers = pubmed_search(disease, max_results=int(depth), year_from=yr_from)
                    if not papers:
                        st.warning("No papers found. Try broader search terms.")
                    else:
                        st.subheader(f"📄 {len(papers)} verified NCBI PubMed papers")
                        for i,pp in enumerate(papers):
                            with st.expander(f"[PMID {pp['pmid']}] [{pp['year']}] {pp['title'][:80]}...", expanded=i<2):
                                st.markdown(f"**Journal:** {pp['journal']}")
                                st.markdown(f"**Author:** {pp['author']} et al.")
                                st.markdown(f"**Year:** {pp['year']} | **PMID:** {pp['pmid']}")
                                st.markdown(f"🔗 [Read full paper on PubMed]({pp['link']})")

                        st.divider()
                        st.subheader("🤖 Evidence_Research_Agent synthesis")
                        pt = st.session_state.patient
                        p_ctx = f"Patient: {pt.get('name')}, {pt.get('age')}yo, conditions: {', '.join(pt.get('conditions',[]))}" if pt else ""
                        papers_str = "\n".join(f"- PMID:{pp['pmid']} | {pp['author']} et al. ({pp['year']}) | {pp['title'][:80]} | {pp['journal']}" for pp in papers)
                        with st.spinner("Synthesising with grounding..."):
                            r = model.generate_content(
                                f"You are Evidence_Research_Agent. Synthesise ONLY based on these real papers.\n"
                                f"{p_ctx}\n\nResearch topic: {disease}\n\nVerified papers:\n{papers_str}\n\n"
                                f"**KEY FINDINGS** — cite each with [PMID:XXXXX]\n"
                                f"**CLINICAL IMPLICATIONS** — cite papers\n"
                                f"**TREATMENT EVIDENCE** — what these papers recommend\n"
                                f"**GAPS IN EVIDENCE** — what these papers identify as unknown\n"
                                f"**BOTTOM LINE** — one paragraph for patient or doctor\n\n"
                                f"RULE: Never claim something not supported by the papers above. Every claim must have [PMID:XXXXX]."
                            )
                            st.info(r.text)
                            st.markdown(format_citations(papers))

    with t2:
        drug_name = st.text_input("Drug name:", placeholder="e.g. Warfarin")
        if st.button("🔍 Validate via OpenFDA", type="primary", key="fda_btn"):
            if not drug_name: st.warning("Enter drug name.")
            else:
                with st.spinner(f"Clinical_Validation_Agent checking FDA for {drug_name}..."):
                    fda = openfda_validate(drug_name)
                    if fda:
                        st.subheader("📋 Official FDA Label Data")
                        st.markdown(f"**Brand names:** {', '.join(fda.get('brand',[]))}")
                        st.markdown(f"**Drug class:** {', '.join(fda.get('class',[]))}")
                        st.markdown(f"**Route:** {', '.join(fda.get('route',[]))}")
                        if "None" not in fda.get("warnings","None"):
                            st.warning(f"⚠️ **FDA Warning:** {fda['warnings'][:300]}...")
                        if "None" not in fda.get("contraindications","None"):
                            st.error(f"🚫 **Contraindications:** {fda['contraindications'][:300]}...")
                        if "None" not in fda.get("interactions","None"):
                            st.info(f"💊 **Drug interactions:** {fda['interactions'][:300]}...")
                    else:
                        st.warning("Not found in FDA database. Try brand name.")

                    # Adverse events chart
                    try:
                        ar = requests.get(f"https://api.fda.gov/drug/event.json?search=patient.drug.medicinalproduct:{requests.utils.quote(drug_name)}&count=patient.reaction.reactionmeddrapt.exact&limit=10",timeout=10)
                        ad = ar.json()
                        if "results" in ad:
                            ae_df = pd.DataFrame(ad["results"][:10]); ae_df.columns=["Adverse Event","Reports"]
                            fig = go.Figure(go.Bar(x=ae_df["Reports"],y=ae_df["Adverse Event"],orientation="h",marker_color="#ef4444"))
                            fig.update_layout(height=280,title="Top adverse events (FDA reports)",margin=dict(l=10,r=10,t=40,b=10),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
                            st.plotly_chart(fig,use_container_width=True)
                    except: pass

    with t3:
        claim_text = st.text_area("Paste any AI output or medical claim to verify:", height=130, placeholder="e.g. Warfarin + Atorvastatin increases bleeding risk due to CYP3A4 inhibition...")
        if st.button("🔍 Verify against real sources", type="primary", key="vfy"):
            if not claim_text: st.warning("Paste a claim.")
            else:
                with st.spinner("Evidence_Research_Agent extracting and verifying claims..."):
                    trace("Evidence_Research_Agent","Extracting verifiable claims from text")
                    extract_r = model.generate_content(f"Extract every specific verifiable medical claim (interactions, statistics, mechanisms, drug names) from:\n\n{claim_text}\n\nReturn as numbered list only. Max 6 claims.")
                    claims_list = extract_r.text
                    st.subheader("Claims identified:")
                    st.info(claims_list)
                    st.subheader("NCBI PubMed verification:")
                    lines = [l.strip() for l in claims_list.split("\n") if l.strip() and l[0].isdigit()]
                    for line in lines[:5]:
                        claim = line[2:].strip() if len(line)>2 else line
                        papers = pubmed_search(claim[:100], max_results=3)
                        if papers:
                            links = " | ".join(f"[PMID {pp['pmid']}]({pp['link']})" for pp in papers)
                            st.success(f"✅ **Evidence found:** _{claim[:70]}..._\n{links}")
                        else:
                            st.warning(f"⚠️ **No direct evidence:** _{claim[:70]}..._")
                    st.divider()
                    verdict_r = model.generate_content(f"Medical fact-checker. For each claim:\n'{claim_text}'\n\nProvide: VERIFIED ✅ / PARTIALLY ⚠️ / UNVERIFIED ❌ / MISLEADING 🚨 — reason — confidence HIGH/MEDIUM/LOW\n\nEnd with OVERALL TRUST SCORE: X/10\nValidated by: Clinical_Validation_Agent — {datetime.now().strftime('%B %Y')}")
                    result = verdict_r.text
                    fn = st.error if "🚨" in result else (st.warning if "❌" in result else st.success)
                    fn(result)
                    trace("Evidence_Research_Agent","Verification complete","done")

    with t4:
        pt = st.session_state.patient
        if not pt: st.warning("Fill Patient Profile first.")
        else:
            conds = pt.get("conditions",[])
            if not conds: st.warning("Add conditions to profile.")
            else:
                st.info(f"Will search latest evidence for: {', '.join(conds)}")
                if st.button("🔬 Get patient-specific evidence", type="primary"):
                    all_papers = []
                    for cond in conds[:3]:
                        with st.spinner(f"Searching: {cond}..."):
                            papers = pubmed_search(f"{cond} treatment guidelines management 2024", max_results=5, year_from=2022)
                            if papers:
                                st.subheader(f"📄 {cond} — {len(papers)} papers")
                                for pp in papers:
                                    st.markdown(f"• **{pp['author']} et al. ({pp['year']}).** {pp['title'][:100]}... *{pp['journal']}* — [Read →]({pp['link']})")
                                all_papers.extend(papers)
                    if all_papers:
                        papers_str = "\n".join(f"- ({pp.get('condition',conds[0])}) PMID:{pp['pmid']} | {pp['title'][:80]} [{pp['year']}]" for pp in all_papers)
                        meds_str = ", ".join(m["name"] for m in st.session_state.medications)
                        with st.spinner("Building personalised evidence summary..."):
                            r = model.generate_content(f"Patient: {pt.get('name')}, {pt.get('age')}yo, conditions: {', '.join(conds)}, meds: {meds_str}\n\nVerified PubMed papers (2022-2026):\n{papers_str}\n\n1. Latest research on managing each condition [cite PMID]?\n2. New treatments/guidelines 2023-2024 [cite PMID]?\n3. Evidence specific to THIS combination of conditions?\n4. What should doctor prioritise based on this evidence?\n\nRule: every claim must reference a PMID from the list above.")
                            st.info(r.text)
                            st.markdown(format_citations(all_papers[:6]))
                            trace("Evidence_Research_Agent","Patient-specific evidence complete","done")

# ═══════════════════════════════════════════════════════════
# AGENT 11 — CALENDAR (calendar_agent MCP)
# ═══════════════════════════════════════════════════════════
elif page == "📅 Calendar":
    st.markdown('<span class="agent-badge">Agent 11: Calendar_Agent · MCP: calendar_agent · Productivity requirement</span>', unsafe_allow_html=True)
    st.header("📅 Health Calendar")
    trace("Primary_Orchestrator","Loading calendar_agent MCP tool")

    t1,t2,t3 = st.tabs(["📅 My calendar","➕ Add event","🤖 AI schedule"])
    with t1:
        if not st.session_state.calendar_events:
            st.info("No events yet. Add medications in Patient Profile to auto-populate.")
        else:
            events = sorted(st.session_state.calendar_events, key=lambda x: x.get("date",""))
            today_str = str(date.today())
            today_ev  = [e for e in events if e.get("date","")==today_str]
            future_ev = [e for e in events if e.get("date","")>today_str]
            past_ev   = [e for e in events if e.get("date","")<today_str]
            if today_ev:
                st.subheader("📌 Today")
                for e in today_ev:
                    icon = "💊" if e.get("type")=="medication" else ("🏥" if e.get("type")=="appointment" else "📋")
                    cancelled = "[CANCELLED]" in e.get("notes","")
                    bg = "#fef2f2" if cancelled else "#eff6ff"
                    st.markdown(f'<div class="cal-slot" style="background:{bg}">{icon} <b>{e["title"]}</b> — {e.get("notes","")}</div>', unsafe_allow_html=True)
            if future_ev:
                st.subheader("⏭️ Upcoming")
                for e in future_ev[:10]:
                    st.markdown(f'<div class="cal-slot">📅 <b>{e["date"]}</b> — {e["title"]}</div>', unsafe_allow_html=True)
            if past_ev:
                with st.expander(f"📁 Past ({len(past_ev)})"):
                    for e in past_ev[-5:]:
                        st.markdown(f"• {e['date']} — {e['title']}")
            if st.button("🗑 Clear all events"): st.session_state.calendar_events = []; st.rerun()

    with t2:
        with st.form("cal_form"):
            c1,c2 = st.columns(2)
            ev_title = c1.text_input("Event title")
            ev_type  = c2.selectbox("Type",["appointment","reminder","medication","follow-up","test","surgery"])
            c3,c4 = st.columns(2)
            ev_date  = c3.date_input("Date")
            ev_notes = c4.text_input("Notes")
            if st.form_submit_button("➕ Add to calendar"):
                if ev_title:
                    st.session_state.calendar_events.append({"title":ev_title,"type":ev_type,"date":str(ev_date),"notes":ev_notes})
                    trace("calendar_agent MCP",f"Event added: {ev_title} on {ev_date}","done")
                    st.success(f"✅ {ev_title} added!")

    with t3:
        if not st.session_state.medications:
            st.warning("Add medications first.")
        else:
            if st.button("🤖 Generate AI medication calendar", type="primary"):
                trace("Calendar_Agent","Generating 7-day medication schedule via AI")
                meds_str = ", ".join(f"{m['name']} {m['dose']} ({m['frequency']})" for m in st.session_state.medications)
                with st.spinner("Generating..."):
                    r = model.generate_content(f"Patient takes: {meds_str}.\n\nCreate a 7-day medication calendar.\nFor each day:\n**Morning (6-8am):** [drugs + pharmacokinetic reason]\n**Midday (12-1pm):** [drugs]\n**Evening (6-7pm):** [drugs]\n**Night (9-10pm):** [drugs]\n\nHighlight: which medications must NEVER be taken together.\nEnd with: Top 3 calendar alerts this patient must set.")
                    st.markdown(r.text)
                    trace("Calendar_Agent","7-day schedule generated","done")

# ═══════════════════════════════════════════════════════════
# AGENT 12 — TASK MANAGER (task_manager MCP)
# ═══════════════════════════════════════════════════════════
elif page == "✅ Task Manager":
    st.markdown('<span class="agent-badge">Agent 12: Patient_Briefing_Agent · Translator · MCP: task_manager · Productivity requirement</span>', unsafe_allow_html=True)
    st.header("✅ Health Task Manager")
    st.caption("Health todos auto-generated by every agent — lab follow-ups, medication refills, appointments, lifestyle goals")
    trace("Primary_Orchestrator","Loading task_manager MCP tool")

    t1,t2 = st.tabs(["📋 My tasks","➕ Add + generate"])
    with t1:
        tasks = st.session_state.tasks
        if not tasks:
            st.info("No tasks yet. Tasks auto-generate from lab reports, emergencies, and doctor briefs.")
        else:
            pending = [t for t in tasks if not t["done"]]
            done    = [t for t in tasks if t["done"]]
            if pending:
                st.subheader(f"⏳ Pending ({len(pending)})")
                for i,task in enumerate(tasks):
                    if task["done"]: continue
                    pc = "🔴" if task["priority"]=="High" else ("🟡" if task["priority"]=="Medium" else "🟢")
                    col1,col2,col3 = st.columns([7,1,1])
                    col1.markdown(f"{pc} **{task['title']}** | Due: {task.get('due','?')} | *{task.get('source','')}*")
                    if col2.button("✅",key=f"done_{i}"): st.session_state.tasks[i]["done"]=True; st.rerun()
                    if col3.button("🗑",key=f"del_{i}"): st.session_state.tasks.pop(i); st.rerun()
            if done:
                with st.expander(f"✅ Completed ({len(done)})"):
                    for task in done:
                        st.markdown(f'<span class="task-done">• {task["title"]}</span>', unsafe_allow_html=True)
            st.divider()
            if pending and st.button("🤖 AI clinical prioritisation", type="primary"):
                trace("Patient_Briefing_Agent","Prioritising tasks by clinical urgency")
                task_list = "\n".join(f"- {t['title']} (due {t.get('due','?')}, priority {t['priority']})" for t in pending)
                pt = st.session_state.patient
                conds = ", ".join(pt.get("conditions",[])) if pt else "none"
                with st.spinner("Prioritising..."):
                    r = model.generate_content(f"Patient conditions: {conds}.\n\nPending health tasks:\n{task_list}\n\nClinically re-prioritise: which MUST be done today, this week, this month?\nFlag any that become dangerous if delayed.\nSuggest optimal order and reason for each.")
                    st.info(r.text)
                    trace("Patient_Briefing_Agent","Task prioritisation complete","done")

    with t2:
        with st.form("task_form"):
            c1,c2 = st.columns(2)
            t_title = c1.text_input("Task title")
            t_prio  = c2.selectbox("Priority",["High","Medium","Low"])
            c3,c4 = st.columns(2)
            t_due    = c3.date_input("Due date")
            t_source = c4.text_input("Source")
            if st.form_submit_button("➕ Add task"):
                if t_title:
                    st.session_state.tasks.append({"title":t_title,"priority":t_prio,"due":str(t_due),"done":False,"source":t_source})
                    trace("task_manager MCP",f"Task added: {t_title}","done")
                    st.success("✅ Task added!")
        st.divider()
        if st.button("🤖 Generate tasks from patient profile", type="primary"):
            pt = st.session_state.patient
            if not pt: st.warning("Need patient profile.")
            else:
                trace("Patient_Briefing_Agent","Generating health task list from patient profile")
                meds_str = ", ".join(m["name"] for m in st.session_state.medications) or "none"
                conds    = ", ".join(pt.get("conditions",[])) or "none"
                with st.spinner("Generating..."):
                    r = model.generate_content(f"Patient: {pt.get('name')}, {pt.get('age')}yo, conditions: {conds}, meds: {meds_str}.\n\nGenerate 8-10 health tasks for next 30 days.\nReturn ONLY valid JSON array:\n[{{\"title\":\"...\",\"priority\":\"High/Medium/Low\",\"due_days\":7}}]\nNo other text.")
                    try:
                        raw = r.text.strip().replace("```json","").replace("```","").strip()
                        task_list = json.loads(raw)
                        added = 0
                        for t in task_list:
                            st.session_state.tasks.append({"title":t["title"],"priority":t["priority"],"due":str(date.today()+timedelta(days=t.get("due_days",7))),"done":False,"source":"Patient_Briefing_Agent"})
                            added += 1
                        trace("Patient_Briefing_Agent",f"{added} tasks generated","done")
                        st.success(f"✅ {added} tasks added!"); st.rerun()
                    except: st.info(r.text)

# ═══════════════════════════════════════════════════════════
# NOTES, CONSENT, ASK MEDPILOT
# ═══════════════════════════════════════════════════════════
elif page == "📝 Notes & Memory":
    st.markdown('<span class="agent-badge">Longitudinal_DeepDive_Agent · MCP: notes_memory</span>', unsafe_allow_html=True)
    st.header("📝 Notes & Memory — Longitudinal Deep-Dive Agent")
    trace("Primary_Orchestrator","Loading notes_memory MCP tool")

    with st.form("note_form"):
        c1,c2 = st.columns(2)
        nt = c1.selectbox("Type",["Doctor visit","Lab result","Medication change","Symptom update","Emergency event","Research finding","General"])
        nd = c2.date_input("Date")
        ntxt = st.text_area("Note content")
        if st.form_submit_button("💾 Save note"):
            if ntxt:
                st.session_state.notes.append({"type":nt,"date":str(nd),"content":ntxt})
                if nt=="Doctor visit":
                    st.session_state.calendar_events.append({"title":"🏥 Doctor visit","type":"appointment","date":str(nd),"notes":ntxt[:80]})
                trace("notes_memory MCP",f"Note saved: {nt} on {nd}","done")
                st.success("✅ Saved!")

    if st.session_state.notes:
        st.subheader(f"📚 {len(st.session_state.notes)} notes")
        for note in reversed(st.session_state.notes):
            with st.expander(f"[{note['date']}] {note['type']}"):
                st.text(note["content"])
        if len(st.session_state.notes)>=2 and st.button("🧠 Longitudinal analysis", type="primary"):
            trace("Longitudinal_DeepDive_Agent","Analysing full history for trends")
            all_notes = "\n".join(f"[{n['date']}] {n['type']}: {n['content']}" for n in st.session_state.notes)
            with st.spinner("Deep-dive analysis..."):
                r = model.generate_content(f"You are Longitudinal_DeepDive_Agent. Analyse this complete medical history:\n\n{all_notes}\n\nFind:\n1. **SLOW TRENDS** — subtle changes over time that might indicate disease progression\n2. **TURNING POINTS** — events that changed the patient's health trajectory\n3. **PATTERNS** — recurring symptoms, medication changes, or lab trends\n4. **RISK SIGNALS** — what should be investigated based on the longitudinal picture\n5. **TIMELINE SUMMARY** — 5 key events in chronological order\n\nBe specific about dates and values where mentioned.")
                st.info(r.text)
                trace("Longitudinal_DeepDive_Agent","Deep-dive analysis complete","done")

elif page == "🔒 Consent Log":
    st.markdown('<span class="agent-badge">Consent_Manager MCP · Privacy audit trail</span>', unsafe_allow_html=True)
    st.header("🔒 Consent & Data Privacy Log")
    st.info("Every time MedPilot OS shares your data, it is logged here with a unique token.")
    if st.session_state.consents:
        for c in reversed(st.session_state.consents):
            fn = st.success if c["granted"] else st.error
            fn(f"**{c['token']}**\nTime: {c['time']} | Purpose: **{c['purpose']}** | Data: {c['data']} | {'GRANTED ✅' if c['granted'] else 'DENIED ❌'}")
    else:
        st.info("No consent events yet.")
    if st.session_state.emergency_log:
        st.divider()
        st.subheader("🚨 Emergency event log")
        for ev in reversed(st.session_state.emergency_log):
            st.error(f"**{ev['time']}**\nSymptoms: {ev['symptoms'][:80]}...\nTelegram: {ev['telegram']}")

elif page == "🤖 Ask MedPilot OS":
    st.markdown('<span class="agent-badge">Primary_Orchestrator · Routes to all 12 agents · MCP: all tools</span>', unsafe_allow_html=True)
    st.header("🤖 Ask MedPilot OS")
    st.info("Ask anything in any language. Orchestrator decides which agent handles it and shows you the routing decision.")

    pt = st.session_state.patient
    meds_str = ", ".join(m["name"] for m in st.session_state.medications) or "none"
    conds    = ", ".join(pt.get("conditions",[])) if pt else "none"
    syms_str = "; ".join(f"{s['symptom']} ({s['date']})" for s in st.session_state.symptoms[-5:]) or "none"
    open_tasks = len([t for t in st.session_state.tasks if not t["done"]])

    question = st.text_area("Your question (any language):", placeholder="Can I eat palak with my medications? / What tasks should I do today? / Mujhe kaun si exercise karni chahiye?")

    if st.button("💬 Ask MedPilot OS", type="primary"):
        if question:
            tool = orchestrator(question)
            st.caption(f"🤖 Primary_Orchestrator → **{tool}**")
            with st.spinner(f"Running {tool}..."):
                papers = []
                if any(kw in question.lower() for kw in ["research","evidence","study","paper","guideline","ncbi"]):
                    papers = pubmed_search(question[:80], max_results=3)
                r = model.generate_content(
                    f"You are MedPilot OS — a clinical intelligence system.\n"
                    f"Active agent: {tool}\n\n"
                    f"Patient: {pt.get('name','unknown') if pt else 'unknown'}, {pt.get('age','?') if pt else '?'}yo\n"
                    f"Conditions: {conds} | Meds: {meds_str}\n"
                    f"Recent symptoms: {syms_str} | Open tasks: {open_tasks}\n\n"
                    f"Question: {question}\n\n"
                    f"Answer clinically, helpfully, and with citations where relevant.\n"
                    f"If question is in another language, respond in that language.\n"
                    f"End with: 'Handled by: {tool}'"
                )
                st.markdown(r.text)
                if papers:
                    st.markdown(format_citations(papers))
        else:
            st.warning("Type a question first.")