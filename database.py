"""
Module 1: Database Module
Handles SQLite storage for traffic logs and anomaly records.
"""

import sqlite3
import json
from datetime import datetime

DB_PATH = "network_anomaly.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    """Create all required tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS traffic_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            src_ip TEXT,
            dst_ip TEXT,
            src_port INTEGER,
            dst_port INTEGER,
            protocol TEXT,
            packet_size INTEGER,
            ttl INTEGER,
            flags TEXT,
            is_anomaly INTEGER DEFAULT 0,
            anomaly_score REAL DEFAULT 0.0
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS anomaly_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            src_ip TEXT,
            dst_ip TEXT,
            protocol TEXT,
            anomaly_score REAL,
            severity TEXT,
            description TEXT,
            raw_features TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS incident_timeline (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            incident_id TEXT,
            timestamp TEXT NOT NULL,
            event_type TEXT,
            src_ip TEXT,
            dst_ip TEXT,
            description TEXT,
            severity TEXT,
            risk_score REAL
        )
    """)

    conn.commit()
    conn.close()
    print("[DB] Database initialized successfully.")


def insert_traffic_log(data: dict):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO traffic_logs 
        (timestamp, src_ip, dst_ip, src_port, dst_port, protocol, packet_size, ttl, flags, is_anomaly, anomaly_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("timestamp", datetime.now().isoformat()),
        data.get("src_ip", ""),
        data.get("dst_ip", ""),
        data.get("src_port", 0),
        data.get("dst_port", 0),
        data.get("protocol", ""),
        data.get("packet_size", 0),
        data.get("ttl", 0),
        data.get("flags", ""),
        int(data.get("is_anomaly", 0)),
        float(data.get("anomaly_score", 0.0)),
    ))
    conn.commit()
    conn.close()


def insert_anomaly_record(data: dict):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO anomaly_records 
        (timestamp, src_ip, dst_ip, protocol, anomaly_score, severity, description, raw_features)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("timestamp", datetime.now().isoformat()),
        data.get("src_ip", ""),
        data.get("dst_ip", ""),
        data.get("protocol", ""),
        float(data.get("anomaly_score", 0.0)),
        data.get("severity", "LOW"),
        data.get("description", ""),
        json.dumps(data.get("raw_features", {})),
    ))
    conn.commit()
    conn.close()


def insert_incident_event(data: dict):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO incident_timeline 
        (incident_id, timestamp, event_type, src_ip, dst_ip, description, severity, risk_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("incident_id", "INC-0001"),
        data.get("timestamp", datetime.now().isoformat()),
        data.get("event_type", "ANOMALY"),
        data.get("src_ip", ""),
        data.get("dst_ip", ""),
        data.get("description", ""),
        data.get("severity", "LOW"),
        float(data.get("risk_score", 0.0)),
    ))
    conn.commit()
    conn.close()


def fetch_recent_traffic(limit=100):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM traffic_logs ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def fetch_recent_anomalies(limit=50):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM anomaly_records ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def fetch_incident_timeline(incident_id=None):
    conn = get_connection()
    cursor = conn.cursor()
    if incident_id:
        cursor.execute(
            "SELECT * FROM incident_timeline WHERE incident_id=? ORDER BY timestamp ASC",
            (incident_id,)
        )
    else:
        cursor.execute("SELECT * FROM incident_timeline ORDER BY timestamp DESC LIMIT 100")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_summary_stats():
    conn = get_connection()
    cursor = conn.cursor()
    stats = {}
    cursor.execute("SELECT COUNT(*) as total FROM traffic_logs")
    stats["total_packets"] = cursor.fetchone()["total"]
    cursor.execute("SELECT COUNT(*) as total FROM traffic_logs WHERE is_anomaly=1")
    stats["total_anomalies"] = cursor.fetchone()["total"]
    cursor.execute("SELECT COUNT(*) as total FROM anomaly_records WHERE severity='HIGH'")
    stats["high_severity"] = cursor.fetchone()["total"]
    cursor.execute("SELECT COUNT(*) as total FROM incident_timeline")
    stats["total_events"] = cursor.fetchone()["total"]
    conn.close()
    return stats


if __name__ == "__main__":
    initialize_database()
    print("[DB] Test complete.")