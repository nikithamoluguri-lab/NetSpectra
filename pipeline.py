"""
Module 6: Main Integration Pipeline
Ties together: capture → feature extraction → AI detection → database → incidents.
Run this as a background process alongside the dashboard.
Usage: python pipeline.py
"""

import time
import threading
from datetime import datetime

from database             import initialize_database, insert_traffic_log, insert_anomaly_record
from feature_extractor    import extract_features_from_packet
from ai_model             import load_model, predict_packet
from capture              import start_simulation, stop_simulation
from incident_reconstruction import reconstruct_incidents

# ── State ───────────────────────────────────────────────
_model  = None
_scaler = None
_stats  = {
    "total_processed": 0,
    "total_anomalies": 0,
    "running":         False,
}
_lock = threading.Lock()


def get_stats() -> dict:
    with _lock:
        return dict(_stats)


# ── Packet handler ───────────────────────────────────────

def handle_packet(packet_data: dict):
    global _model, _scaler, _stats

    if _model is None:
        return  # model not loaded yet

    # 1. Run AI prediction
    result = predict_packet(packet_data, _model, _scaler)

    # 2. Merge prediction into packet_data
    packet_data["is_anomaly"]    = result["is_anomaly"]
    packet_data["anomaly_score"] = result["anomaly_score"]

    # 3. Store traffic log
    insert_traffic_log(packet_data)

    with _lock:
        _stats["total_processed"] += 1

    # 4. If anomaly, store anomaly record
    if result["is_anomaly"]:
        insert_anomaly_record({
            "timestamp":    packet_data.get("timestamp", datetime.now().isoformat()),
            "src_ip":       packet_data.get("src_ip", ""),
            "dst_ip":       packet_data.get("dst_ip", ""),
            "protocol":     packet_data.get("protocol", ""),
            "anomaly_score":result["anomaly_score"],
            "severity":     result["severity"],
            "description":  result["description"],
            "raw_features": result["features"],
        })
        with _lock:
            _stats["total_anomalies"] += 1

        print(
            f"[⚠]  ANOMALY | {packet_data.get('src_ip'):>15} → {packet_data.get('dst_ip'):<15} "
            f"| Score={result['anomaly_score']:.3f} | {result['severity']} | {result['description']}"
        )
    else:
        print(
            f"[✓]  NORMAL  | {packet_data.get('src_ip'):>15} → {packet_data.get('dst_ip'):<15}"
        )


# ── Incident reconstruction loop ─────────────────────────

def _incident_loop(interval_seconds: int = 60):
    while _stats["running"]:
        time.sleep(interval_seconds)
        print("[Pipeline] Reconstructing incidents …")
        incidents = reconstruct_incidents()
        print(f"[Pipeline] Found {len(incidents)} incident(s).")


# ── Start / Stop ─────────────────────────────────────────

def start_pipeline(packets_per_second: float = 3.0, attack_ratio: float = 0.10):
    global _model, _scaler

    print("[Pipeline] Initialising database …")
    initialize_database()

    print("[Pipeline] Loading AI model …")
    _model, _scaler = load_model()

    with _lock:
        _stats["running"] = True

    print("[Pipeline] Starting packet capture/simulation …")
    start_simulation(
        callback=handle_packet,
        packets_per_second=packets_per_second,
        attack_ratio=attack_ratio,
    )

    # Start incident reconstruction in background
    inc_thread = threading.Thread(target=_incident_loop, args=(30,), daemon=True)
    inc_thread.start()

    print("[Pipeline] ✅ Pipeline running. Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(5)
            s = get_stats()
            print(
                f"[Pipeline] Stats → Processed: {s['total_processed']} | "
                f"Anomalies: {s['total_anomalies']}"
            )
    except KeyboardInterrupt:
        print("\n[Pipeline] Stopping …")
        stop_simulation()
        with _lock:
            _stats["running"] = False
        print("[Pipeline] Stopped.")


if __name__ == "__main__":
    start_pipeline(packets_per_second=3.0, attack_ratio=0.10)