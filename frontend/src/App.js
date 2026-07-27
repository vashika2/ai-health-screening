import { useState } from "react";
import axios from "axios";

const API = "http://localhost:8000";

export default function App() {
  const [mode, setMode] = useState("tb");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [form, setForm] = useState({
    age: "", symptoms: "", duration: "",
    diabetes_years: "", hba1c: ""
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFile = (e) => {
    const f = e.target.files[0];
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError("");
  };

  const handleSubmit = async () => {
    if (!file) return setError("Please upload an image first.");
    setLoading(true);
    setError("");
    setResult(null);

    const data = new FormData();
    data.append("file", file);
    if (mode === "tb") {
      data.append("age", form.age);
      data.append("symptoms", form.symptoms);
      data.append("duration", form.duration);
    } else {
      data.append("age", form.age);
      data.append("diabetes_years", form.diabetes_years);
      data.append("hba1c", form.hba1c);
    }

    try {
      const res = await axios.post(`${API}/predict/${mode}`, data);
      setResult(res.data);
    } catch (err) {
      setError(err.response?.data?.error || "Something went wrong. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  return (
      <div style={styles.page}>
        <div style={styles.container}>

          {/* Header */}
          <div style={styles.header}>
            <h1 style={styles.title}>🏥 AI Health Screening</h1>
            <p style={styles.subtitle}>Early detection of TB & Diabetic Retinopathy</p>
          </div>

          {/* Mode Toggle */}
          <div style={styles.toggleRow}>
            <button
                onClick={() => { setMode("tb"); setResult(null); setError(""); }}
                style={mode === "tb" ? styles.toggleActive : styles.toggleInactive}
            >
              🫁 TB Detection
            </button>
            <button
                onClick={() => { setMode("dr"); setResult(null); setError(""); }}
                style={mode === "dr" ? styles.toggleActive : styles.toggleInactive}
            >
              👁️ Diabetic Retinopathy
            </button>
          </div>

          <div style={styles.card}>

            {/* Upload */}
            <div style={styles.section}>
              <label style={styles.label}>
                Upload {mode === "tb" ? "Chest X-Ray" : "Retinal Fundus Image"}
              </label>
              <input
                  type="file"
                  accept="image/*"
                  onChange={handleFile}
                  style={styles.fileInput}
              />
              {preview && (
                  <img src={preview} alt="preview" style={styles.preview} />
              )}
            </div>

            {/* Patient Info */}
            <div style={styles.section}>
              <label style={styles.label}>Patient Information (Optional)</label>
              <div style={styles.formGrid}>
                <div>
                  <p style={styles.inputLabel}>Age</p>
                  <input
                      type="text"
                      placeholder="e.g. 45"
                      value={form.age}
                      onChange={e => setForm({ ...form, age: e.target.value })}
                      style={styles.input}
                  />
                </div>

                {mode === "tb" ? (
                    <>
                      <div>
                        <p style={styles.inputLabel}>Symptoms</p>
                        <input
                            type="text"
                            placeholder="e.g. cough, fever"
                            value={form.symptoms}
                            onChange={e => setForm({ ...form, symptoms: e.target.value })}
                            style={styles.input}
                        />
                      </div>
                      <div>
                        <p style={styles.inputLabel}>Duration of Symptoms</p>
                        <input
                            type="text"
                            placeholder="e.g. 3 weeks"
                            value={form.duration}
                            onChange={e => setForm({ ...form, duration: e.target.value })}
                            style={styles.input}
                        />
                      </div>
                    </>
                ) : (
                    <>
                      <div>
                        <p style={styles.inputLabel}>Diabetes Duration (years)</p>
                        <input
                            type="text"
                            placeholder="e.g. 8"
                            value={form.diabetes_years}
                            onChange={e => setForm({ ...form, diabetes_years: e.target.value })}
                            style={styles.input}
                        />
                      </div>
                      <div>
                        <p style={styles.inputLabel}>Last HbA1c (%)</p>
                        <input
                            type="text"
                            placeholder="e.g. 8.2"
                            value={form.hba1c}
                            onChange={e => setForm({ ...form, hba1c: e.target.value })}
                            style={styles.input}
                        />
                      </div>
                    </>
                )}
              </div>
            </div>

            {/* Submit Button */}
            <button
                onClick={handleSubmit}
                disabled={loading}
                style={loading ? styles.btnDisabled : styles.btn}
            >
              {loading ? "⏳ Analyzing... this may take 30 seconds" : "🔍 Analyze Image"}
            </button>

            {/* Error */}
            {error && (
                <div style={styles.errorBox}>
                  ❌ {error}
                </div>
            )}

            {/* Result — everything below only renders once result exists */}
            {result && (
                <div style={result.prediction.needs_referral ? styles.resultDanger : styles.resultSafe}>

                  <h2 style={styles.resultTitle}>
                    {result.prediction.needs_referral
                        ? "⚠️ Referral Recommended"
                        : "✅ No Immediate Referral Needed"}
                  </h2>

                  <p style={styles.resultLabel}>{result.prediction.label}</p>
                  <p style={styles.resultMeta}>
                    Confidence: <strong>{result.prediction.confidence}%</strong>
                  </p>

                  {result.prediction.grade !== undefined && (
                      <p style={styles.resultMeta}>
                        DR Grade: <strong>{result.prediction.grade} / 4</strong>
                      </p>
                  )}

                  {result.prediction.severity && (
                      <p style={styles.resultMeta}>
                        Severity: <strong>{result.prediction.severity}</strong>
                      </p>
                  )}

                  {/* Grad-CAM Heatmap */}
                  {result.heatmap && (
                      <div style={styles.heatmapBox}>
                        <p style={styles.heatmapLabel}>🔥 AI Focus Heatmap (Grad-CAM)</p>
                        <img
                            src={result.heatmap}
                            alt="Grad-CAM heatmap"
                            style={styles.heatmapImage}
                        />
                        <p style={styles.heatmapCaption}>
                          Red/yellow areas show where the AI focused most when making its prediction
                        </p>
                      </div>
                  )}

                  <hr style={styles.divider} />

                  <h3 style={styles.reportTitle}>📋 AI Generated Report</h3>
                  <p style={styles.reportText}>
                    {result.report.replace(/##/g, "").replace(/\*\*/g, "")}
                  </p>
                </div>
            )}

          </div>
        </div>
      </div>
  );
}

const styles = {
  page: {
    minHeight: "100vh",
    backgroundColor: "#f0f4f8",
    padding: "30px 16px",
    fontFamily: "Segoe UI, sans-serif"
  },
  container: {
    maxWidth: "700px",
    margin: "0 auto"
  },
  header: {
    textAlign: "center",
    marginBottom: "24px"
  },
  title: {
    fontSize: "28px",
    fontWeight: "bold",
    color: "#1a3c5e",
    margin: 0
  },
  subtitle: {
    color: "#666",
    marginTop: "6px"
  },
  toggleRow: {
    display: "flex",
    gap: "12px",
    justifyContent: "center",
    marginBottom: "20px"
  },
  toggleActive: {
    padding: "10px 24px",
    borderRadius: "999px",
    border: "none",
    backgroundColor: "#1a3c5e",
    color: "white",
    fontWeight: "bold",
    cursor: "pointer",
    fontSize: "15px"
  },
  toggleInactive: {
    padding: "10px 24px",
    borderRadius: "999px",
    border: "2px solid #1a3c5e",
    backgroundColor: "white",
    color: "#1a3c5e",
    fontWeight: "bold",
    cursor: "pointer",
    fontSize: "15px"
  },
  card: {
    backgroundColor: "white",
    borderRadius: "16px",
    padding: "28px",
    boxShadow: "0 4px 20px rgba(0,0,0,0.08)"
  },
  section: {
    marginBottom: "20px"
  },
  label: {
    fontWeight: "bold",
    color: "#333",
    display: "block",
    marginBottom: "8px"
  },
  fileInput: {
    width: "100%",
    padding: "8px",
    border: "1px solid #ddd",
    borderRadius: "8px",
    fontSize: "14px"
  },
  preview: {
    marginTop: "12px",
    maxHeight: "220px",
    borderRadius: "8px",
    border: "1px solid #ddd",
    objectFit: "contain"
  },
  formGrid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: "12px"
  },
  inputLabel: {
    fontSize: "13px",
    color: "#555",
    margin: "0 0 4px 0"
  },
  input: {
    width: "100%",
    padding: "8px 10px",
    border: "1px solid #ddd",
    borderRadius: "8px",
    fontSize: "14px",
    boxSizing: "border-box"
  },
  btn: {
    width: "100%",
    padding: "14px",
    backgroundColor: "#1a3c5e",
    color: "white",
    border: "none",
    borderRadius: "12px",
    fontSize: "16px",
    fontWeight: "bold",
    cursor: "pointer",
    marginTop: "8px"
  },
  btnDisabled: {
    width: "100%",
    padding: "14px",
    backgroundColor: "#aaa",
    color: "white",
    border: "none",
    borderRadius: "12px",
    fontSize: "16px",
    fontWeight: "bold",
    cursor: "not-allowed",
    marginTop: "8px"
  },
  errorBox: {
    marginTop: "16px",
    padding: "12px",
    backgroundColor: "#fff0f0",
    border: "1px solid #ffcccc",
    borderRadius: "8px",
    color: "#cc0000",
    fontSize: "14px"
  },
  resultDanger: {
    marginTop: "20px",
    padding: "20px",
    backgroundColor: "#fff5f5",
    border: "1px solid #ffaaaa",
    borderRadius: "12px",
    color: "#7a0000"
  },
  resultSafe: {
    marginTop: "20px",
    padding: "20px",
    backgroundColor: "#f0fff4",
    border: "1px solid #aaddaa",
    borderRadius: "12px",
    color: "#004d1a"
  },
  resultTitle: {
    fontSize: "18px",
    fontWeight: "bold",
    margin: "0 0 8px 0"
  },
  resultLabel: {
    fontSize: "16px",
    fontWeight: "600",
    margin: "4px 0"
  },
  resultMeta: {
    fontSize: "14px",
    margin: "4px 0"
  },
  heatmapBox: {
    marginTop: "16px",
    marginBottom: "8px"
  },
  heatmapLabel: {
    fontSize: "13px",
    fontWeight: "bold",
    marginBottom: "6px"
  },
  heatmapImage: {
    width: "100%",
    maxHeight: "300px",
    objectFit: "contain",
    borderRadius: "8px",
    border: "1px solid rgba(0,0,0,0.1)"
  },
  heatmapCaption: {
    fontSize: "11px",
    color: "#666",
    marginTop: "4px"
  },
  divider: {
    margin: "16px 0",
    border: "none",
    borderTop: "1px solid rgba(0,0,0,0.1)"
  },
  reportTitle: {
    fontSize: "15px",
    fontWeight: "bold",
    margin: "0 0 8px 0"
  },
  reportText: {
    fontSize: "14px",
    lineHeight: "1.7",
    whiteSpace: "pre-wrap",
    margin: 0
  }
};