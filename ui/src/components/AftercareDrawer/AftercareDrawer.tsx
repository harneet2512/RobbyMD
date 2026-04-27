import { useState } from "react";
import styles from "./AftercareDrawer.module.css";

const API_BASE = `http://${window.location.hostname}:8420`;

interface FollowUp {
  action: string;
  timeframe: string | null;
}

interface MedInstruction {
  medication: string;
  instructions: string;
}

interface AftercarePackage {
  encounter_id: string;
  summary: string;
  follow_up_plan: FollowUp[];
  medication_instructions: MedInstruction[];
  red_flag_symptoms: string[];
  medical_terms: string[];
  approved: boolean;
  approval_note: string | null;
}

interface Props {
  encounterId: string;
  onClose: () => void;
}

export function AftercareDrawer({ encounterId, onClose }: Props) {
  const [pkg, setPkg] = useState<AftercarePackage | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [approving, setApproving] = useState(false);

  async function handleGenerate() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/aftercare/${encounterId}/generate`, {
        method: "POST",
      });
      if (!res.ok) throw new Error(`${res.status}`);
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setPkg(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to generate");
    } finally {
      setLoading(false);
    }
  }

  async function handleApprove() {
    setApproving(true);
    try {
      const res = await fetch(`${API_BASE}/api/aftercare/${encounterId}/approve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ approval_note: null }),
      });
      if (!res.ok) throw new Error(`${res.status}`);
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setPkg(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to approve");
    } finally {
      setApproving(false);
    }
  }

  return (
    <div className={styles.overlay} onClick={onClose}>
      <aside className={styles.drawer} onClick={(e) => e.stopPropagation()}>
        <header className={styles.header}>
          <h2 className={styles.title}>Patient Aftercare Package</h2>
          <button className={styles.closeBtn} onClick={onClose} type="button" aria-label="Close">
            ×
          </button>
        </header>

        {!pkg && !loading && (
          <div className={styles.generatePrompt}>
            <p className={styles.desc}>
              Generate a patient-facing aftercare package from the current encounter state.
              You will review and approve before it is finalized.
            </p>
            <button className={styles.generateBtn} onClick={handleGenerate} type="button">
              Generate Aftercare Package
            </button>
          </div>
        )}

        {loading && (
          <div className={styles.loading}>Generating from encounter state...</div>
        )}

        {error && (
          <div className={styles.error}>{error}</div>
        )}

        {pkg && (
          <div className={styles.package}>
            <section className={styles.section}>
              <h3 className={styles.sectionTitle}>Visit Summary</h3>
              <p className={styles.sectionBody}>{pkg.summary}</p>
            </section>

            {pkg.medication_instructions.length > 0 && (
              <section className={styles.section}>
                <h3 className={styles.sectionTitle}>Medications</h3>
                {pkg.medication_instructions.map((m, i) => (
                  <div key={i} className={styles.medItem}>
                    <span className={styles.medName}>{m.medication}</span>
                    <span className={styles.medInstructions}>{m.instructions}</span>
                  </div>
                ))}
              </section>
            )}

            {pkg.follow_up_plan.length > 0 && (
              <section className={styles.section}>
                <h3 className={styles.sectionTitle}>Follow-Up Plan</h3>
                {pkg.follow_up_plan.map((f, i) => (
                  <div key={i} className={styles.followUpItem}>
                    <span>{f.action}</span>
                    {f.timeframe && <span className={styles.timeframe}>{f.timeframe}</span>}
                  </div>
                ))}
              </section>
            )}

            {pkg.red_flag_symptoms.length > 0 && (
              <section className={styles.section}>
                <h3 className={styles.sectionTitle}>Red Flag Symptoms</h3>
                <p className={styles.redFlagNote}>Return to the emergency department if you experience:</p>
                <ul className={styles.redFlagList}>
                  {pkg.red_flag_symptoms.map((s, i) => (
                    <li key={i} className={styles.redFlagItem}>{s}</li>
                  ))}
                </ul>
              </section>
            )}

            {pkg.medical_terms.length > 0 && (
              <section className={styles.section}>
                <h3 className={styles.sectionTitle}>Medical Terms Explained</h3>
                <div className={styles.termsList}>
                  {pkg.medical_terms.map((t, i) => (
                    <span key={i} className={styles.term}>{t}</span>
                  ))}
                </div>
              </section>
            )}

            <footer className={styles.footer}>
              {pkg.approved ? (
                <div className={styles.approvedBadge}>Approved</div>
              ) : (
                <button
                  className={styles.approveBtn}
                  onClick={handleApprove}
                  disabled={approving}
                  type="button"
                >
                  {approving ? "Approving..." : "Approve & Send to Patient"}
                </button>
              )}
            </footer>
          </div>
        )}
      </aside>
    </div>
  );
}
