# medpilot
MedPilot OS — A 12-agent MCP-powered clinical intelligence system for medication safety, emergency response, symptom tracking &amp; evidence-based care. Built with Gemini 2.5 Flash + PubMed + FDA APIs.
# 🏥 MedPilot OS
### 12-Agent · MCP-Powered · Clinical Intelligence System

## 🚀 Features
- 👤 Patient Profile & Medication Management
- 📸 Prescription OCR + FDA Validation
- 💊 Polypharmacy Safety Matrix
- ⏱️ Symptom Trajectory (FAST/Sepsis/DKA detection)
- 🍎 Food-Drug Interaction Scanner
- 🚨 Emergency Cascade with Telegram Alerts
- 🔬 Real PubMed Evidence Research
- 👨‍⚕️ SBAR Physician Brief Generator
- 📅 Health Calendar & Task Manager

## ⚙️ Run Locally
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"
streamlit run app.py

## 🛠️ Tech Stack
Streamlit · Google Gemini 2.5 Flash · Plotly
NCBI PubMed API · OpenFDA · NIH RxNav · Telegram
