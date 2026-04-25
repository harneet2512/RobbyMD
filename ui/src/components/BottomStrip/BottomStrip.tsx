import { sendClose } from "@/api/socket";
import styles from "./BottomStrip.module.css";

export function BottomStrip() {
  return (
    <div className={styles.strip}>
      <span className={styles.disclaimer}>
        research prototype · not a medical device · physician decides
      </span>

      <span className={styles.priorVisits}>· prior visits ·</span>

      <button className={styles.closeBtn} onClick={sendClose}>
        close encounter
      </button>
    </div>
  );
}
