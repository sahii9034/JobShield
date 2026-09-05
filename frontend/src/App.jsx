import { useState } from "react";
import "./App.css";

function App() {
  const [inputMode, setInputMode] = useState("url");
  const [jobUrl, setJobUrl] = useState("");
  const [jobText, setJobText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const analyzeJob = async () => {
    if (inputMode === "url" && !jobUrl.trim()) {
      setError("Please enter a job posting URL.");
      return;
    }

    if (inputMode === "text" && !jobText.trim()) {
      setError("Please paste a job description first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const requestBody =
        inputMode === "url"
          ? { job_url: jobUrl.trim() }
          : { job_text: jobText.trim() };

      const response = await fetch(
        "http://127.0.0.1:8000/analyze",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestBody),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Unable to analyze the job."
        );
      }

      setResult(data);

    } catch (err) {
      setError(
        err.message ||
          "Could not connect to JobShield. Make sure the FastAPI server is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const score = result
    ? (result.fraud_probability * 100).toFixed(1)
    : "0.0";

  return (
    <div className="app">

      {/* ============================================
          NAVBAR
      ============================================ */}

      <header className="navbar">

        <div className="brand">

          <div className="brand-icon">
            🛡
          </div>

          <span>
            JobShield
          </span>

        </div>

        <span className="tagline">
          AI-powered job safety analysis
        </span>

      </header>


      <main className="container">

        {/* ============================================
            HERO
        ============================================ */}

        <section className="hero">

          <p className="eyebrow">
            JOB SCAM DETECTION
          </p>

          <h1>
            Is this job
            <span> safe to pursue?</span>
          </h1>

          <p className="hero-text">
            Analyze a publicly accessible job posting using
            machine learning and an explainable AI layer.
          </p>

        </section>


        {/* ============================================
            INPUT
        ============================================ */}

        <section className="input-card">

          <div className="input-header">

            <div>

              <h2>
                Analyze a job posting
              </h2>

              <p>
                Enter a public job URL or paste the job description.
              </p>

            </div>

            <span className="input-label">
              JOB INPUT
            </span>

          </div>


          {/* Input mode tabs */}

          <div className="input-tabs">

            <button
              className={
                inputMode === "url"
                  ? "active"
                  : ""
              }
              onClick={() => {
                setInputMode("url");
                setError("");
                setResult(null);
              }}
            >
              Job URL
            </button>

            <button
              className={
                inputMode === "text"
                  ? "active"
                  : ""
              }
              onClick={() => {
                setInputMode("text");
                setError("");
                setResult(null);
              }}
            >
              Job Description
            </button>

          </div>


          {/* URL or text input */}

          {inputMode === "url" ? (

            <input
              className="url-input"
              type="url"
              value={jobUrl}
              onChange={(e) =>
                setJobUrl(e.target.value)
              }
              placeholder="https://www.example.com/jobs/software-engineer"
            />

          ) : (

            <textarea
              value={jobText}
              onChange={(e) =>
                setJobText(e.target.value)
              }
              placeholder="Paste the job title, company information, description, requirements, salary details, and other available information here..."
            />

          )}


          {/* Input footer */}

          <div className="input-footer">

            <span>
              {inputMode === "url"
                ? "Publicly accessible job URLs only"
                : `${jobText.length.toLocaleString()} characters`}
            </span>

            <button
              onClick={analyzeJob}
              disabled={loading}
            >
              {loading
                ? "Analyzing..."
                : "Analyze Job →"}
            </button>

          </div>


          {/* Error */}

          {error && (
            <div className="error">
              {error}
            </div>
          )}

        </section>


        {/* ============================================
            RESULTS
        ============================================ */}

        {result && (

          <section className="results">

            {/* Result header */}

            <div className="result-header">

              <div>

                <p className="eyebrow">
                  ANALYSIS RESULT
                </p>

                <h2>
                  {result.prediction}
                </h2>

              </div>

              <div
                className={`risk-badge ${
                  result.risk_level.toLowerCase()
                }`}
              >
                {result.risk_level} Risk
              </div>

            </div>


            {/* ========================================
                SCORE
            ======================================== */}

            <div className="score-card">

              <div className="score-number">
                {score}%
              </div>

              <div>

                <h3>
                  Model Score
                </h3>

                <p>
                  Score produced by the trained
                  machine-learning model.
                </p>

              </div>

            </div>


            {/* ========================================
                AI EXPLANATION + MODEL SIGNALS
            ======================================== */}

            <div className="result-grid">


              {/* AI Explanation */}

              <div className="panel">

                <div className="panel-title">
                  <span>
                    AI Explanation
                  </span>
                </div>

                <div className="explanation">
                  {result.explanation}
                </div>

              </div>


              {/* Model Signals */}

              <div className="panel">

                <div className="panel-title">
                  <span>
                    Model Signals
                  </span>
                </div>

                <div className="signals">

                  {result.model_signals &&
                    result.model_signals
                      .filter(
                        (item) =>
                          Math.abs(
                            item.contribution
                          ) > 0.03
                      )
                      .slice(0, 5)
                      .map((item) => (

                        <div
                          className={`signal ${
                            result.prediction ===
                            "Potentially Fraudulent"
                              ? "fraud"
                              : "legitimate"
                          }`}
                          key={item.feature}
                        >

                          <span>
                            {result.prediction ===
                            "Potentially Fraudulent"
                              ? "+"
                              : "−"}
                          </span>

                          <span>
                            {item.feature}
                          </span>

                        </div>

                      ))}

                </div>

              </div>

            </div>


            {/* ========================================
                COUNTER SIGNALS
            ======================================== */}

            <div className="panel counter-panel">

              <div className="panel-title">

                <span>
                  Counter-signals
                </span>

              </div>

              <div className="signals">

                {result.counter_signals &&
                  result.counter_signals
                    .filter(
                      (item) =>
                        Math.abs(
                          item.contribution
                        ) > 0.03
                    )
                    .slice(0, 5)
                    .map((item) => (

                      <div
                        className={`signal ${
                          result.prediction ===
                          "Potentially Fraudulent"
                            ? "legitimate"
                            : "fraud"
                        }`}
                        key={item.feature}
                      >

                        <span>
                          {result.prediction ===
                          "Potentially Fraudulent"
                            ? "−"
                            : "+"
                          }
                        </span>

                        <span>
                          {item.feature}
                        </span>

                      </div>

                    ))}

              </div>

            </div>


            {/* ========================================
                CAUTION
            ======================================== */}

            <div className="caution">

              <strong>
                Important:
              </strong>{" "}

              This model prediction is not proof that
              a job is fraudulent or legitimate.
              Individual features do not establish
              fraud on their own.

            </div>

          </section>

        )}


        {/* ============================================
            FOOTER
        ============================================ */}

        <footer>

          <p>
            JobShield uses a machine-learning model trained
            on historical job-posting data. Results should
            be used as an additional screening signal, not
            as a definitive judgment.
          </p>

        </footer>

      </main>

    </div>
  );
}

export default App;