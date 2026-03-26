"""
Module 5: Incident Reconstruction Module
Groups anomalies into incidents and builds attack timelines.
"""

import uuid
from datetime import datetime, timedelta
from database import (
    fetch_recent_anomalies,
    fetch_incident_timeline,
    insert_incident_event,
)


def compute_risk_score(anomaly: dict) -> float:
    """
    Compute a 0-100 risk score for a single anomaly record.
    """
    score = 0.0
    base  = float(anomaly.get("anomaly_score", 0)) * 50  # 0-50 from AI score
    score += base

    severity = anomaly.get("severity", "LOW")
    if severity == "HIGH":
        score += 40
    elif severity == "MEDIUM":
        score += 20
    else:
        score += 5

    # Suspicious destination ports
    dst_port = int(anomaly.get("dst_port", 0) or 0)
    if dst_port in [22, 23, 3389, 445, 1433, 3306]:
        score += 10

    return min(round(score, 2), 100.0)


def classify_attack_type(anomaly: dict) -> str:
    """Heuristically classify the likely attack type."""
    desc    = (anomaly.get("description") or "").lower()
    proto   = (anomaly.get("protocol")    or "").upper()
    score   = float(anomaly.get("anomaly_score", 0))

    if "port" in desc and score > 0.6:
        return "Port Scan"
    if "oversized" in desc:
        return "Possible DDoS / Flood"
    if "low ttl" in desc:
        return "TTL Manipulation / Traceroute"
    if "rst+syn" in desc:
        return "TCP Flag Abuse"
    if proto == "ICMP":
        return "ICMP Probe"
    if score > 0.8:
        return "High-Confidence Intrusion Attempt"
    return "Unknown Anomaly"


def reconstruct_incidents(window_minutes: int = 5) -> list[dict]:
    """
    Fetch recent anomalies, group them into incidents by source IP
    within a time window, and write events to incident_timeline table.
    Returns a list of incident summaries.
    """
    anomalies = fetch_recent_anomalies(limit=200)
    if not anomalies:
        return []

    # Group by source IP
    groups: dict[str, list] = {}
    for a in anomalies:
        key = a.get("src_ip", "unknown")
        groups.setdefault(key, []).append(a)

    incidents = []

    for src_ip, events in groups.items():
        if len(events) < 2:
            continue  # single event — not an incident

        incident_id   = f"INC-{str(uuid.uuid4())[:8].upper()}"
        total_risk    = sum(compute_risk_score(e) for e in events) / len(events)
        attack_types  = list({classify_attack_type(e) for e in events})
        start_time    = min(e.get("timestamp", "") for e in events)
        end_time      = max(e.get("timestamp", "") for e in events)

        severity = "HIGH" if total_risk > 60 else ("MEDIUM" if total_risk > 30 else "LOW")

        # Write each event to timeline
        for evt in events:
            insert_incident_event({
                "incident_id": incident_id,
                "timestamp":   evt.get("timestamp", datetime.now().isoformat()),
                "event_type":  classify_attack_type(evt),
                "src_ip":      evt.get("src_ip", ""),
                "dst_ip":      evt.get("dst_ip", ""),
                "description": evt.get("description", ""),
                "severity":    evt.get("severity", "LOW"),
                "risk_score":  compute_risk_score(evt),
            })

        summary = {
            "incident_id":   incident_id,
            "src_ip":        src_ip,
            "event_count":   len(events),
            "attack_types":  attack_types,
            "avg_risk_score": round(total_risk, 2),
            "severity":      severity,
            "start_time":    start_time,
            "end_time":      end_time,
        }
        incidents.append(summary)

    return incidents


def generate_incident_report(incident_id: str) -> str:
    """Generate a text report for an incident."""
    events = fetch_incident_timeline(incident_id)
    if not events:
        return f"No events found for incident {incident_id}."

    lines = [
        f"═══════════════════════════════════════════",
        f"  INCIDENT REPORT — {incident_id}",
        f"═══════════════════════════════════════════",
        f"  Total Events : {len(events)}",
        f"  Source IP    : {events[0].get('src_ip', 'N/A')}",
        f"  Time Range   : {events[0].get('timestamp','')} → {events[-1].get('timestamp','')}",
        f"",
        f"  TIMELINE:",
    ]

    for i, ev in enumerate(events, 1):
        lines.append(
            f"  [{i:02d}] {ev['timestamp'][:19]} | {ev['event_type']:30s} | "
            f"Risk={ev['risk_score']:5.1f} | {ev['severity']}"
        )

    lines.append(f"═══════════════════════════════════════════")
    return "\n".join(lines)


if __name__ == "__main__":
    from database import initialize_database
    initialize_database()
    incidents = reconstruct_incidents()
    if incidents:
        for inc in incidents:
            print(generate_incident_report(inc["incident_id"]))
    else:
        print("[Incident] No multi-event incidents found yet. Run the main pipeline first.")