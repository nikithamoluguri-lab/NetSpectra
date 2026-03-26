"""
Module 2: Feature Extraction Module
Extracts structured features from raw packets for AI analysis.
"""

import random
import socket
from datetime import datetime

# Protocol number to name mapping
PROTOCOL_MAP = {
    1: "ICMP",
    6: "TCP",
    17: "UDP",
    58: "ICMPv6",
}

FEATURE_COLUMNS = [
    "packet_size",
    "src_port",
    "dst_port",
    "ttl",
    "protocol_num",
    "is_tcp",
    "is_udp",
    "is_icmp",
    "has_syn",
    "has_ack",
    "has_fin",
    "has_rst",
    "is_well_known_port",
    "is_private_src",
    "is_private_dst",
]


def ip_is_private(ip: str) -> int:
    """Returns 1 if IP is in private range, 0 otherwise."""
    try:
        packed = socket.inet_aton(ip)
        b = list(packed)
        if b[0] == 10:
            return 1
        if b[0] == 172 and 16 <= b[1] <= 31:
            return 1
        if b[0] == 192 and b[1] == 168:
            return 1
    except Exception:
        pass
    return 0


def extract_features_from_packet(packet_data: dict) -> dict:
    """
    Given a packet dictionary (from scapy or simulated),
    returns a flat feature dict ready for ML inference.
    """
    flags = packet_data.get("flags", "")
    protocol = packet_data.get("protocol", "TCP")
    proto_num = {"TCP": 6, "UDP": 17, "ICMP": 1}.get(protocol.upper(), 0)

    features = {
        "packet_size":        int(packet_data.get("packet_size", 0)),
        "src_port":           int(packet_data.get("src_port", 0)),
        "dst_port":           int(packet_data.get("dst_port", 0)),
        "ttl":                int(packet_data.get("ttl", 64)),
        "protocol_num":       proto_num,
        "is_tcp":             1 if protocol.upper() == "TCP" else 0,
        "is_udp":             1 if protocol.upper() == "UDP" else 0,
        "is_icmp":            1 if protocol.upper() == "ICMP" else 0,
        "has_syn":            1 if "S" in flags else 0,
        "has_ack":            1 if "A" in flags else 0,
        "has_fin":            1 if "F" in flags else 0,
        "has_rst":            1 if "R" in flags else 0,
        "is_well_known_port": 1 if int(packet_data.get("dst_port", 0)) < 1024 else 0,
        "is_private_src":     ip_is_private(packet_data.get("src_ip", "")),
        "is_private_dst":     ip_is_private(packet_data.get("dst_ip", "")),
    }
    return features


def simulate_packet(is_attack: bool = False) -> dict:
    """
    Simulates a network packet (used when live capture is unavailable).
    Returns a packet dict similar to what Scapy would produce.
    """
    protocols = ["TCP", "UDP", "ICMP"]
    timestamp = datetime.now().isoformat()

    if is_attack:
        # Simulate attack-like characteristics
        return {
            "timestamp":   timestamp,
            "src_ip":      f"{random.randint(1,254)}.{random.randint(0,254)}.{random.randint(0,254)}.{random.randint(1,254)}",
            "dst_ip":      f"192.168.{random.randint(0,5)}.{random.randint(1,10)}",
            "src_port":    random.randint(40000, 65535),
            "dst_port":    random.choice([22, 23, 3389, 445, 8080]),
            "protocol":    "TCP",
            "packet_size": random.randint(1400, 1500),  # high size
            "ttl":         random.randint(1, 30),        # very low TTL
            "flags":       random.choice(["S", "RS", "SF", ""]),
        }
    else:
        # Simulate normal traffic
        return {
            "timestamp":   timestamp,
            "src_ip":      f"192.168.{random.randint(0,2)}.{random.randint(1,50)}",
            "dst_ip":      f"10.0.{random.randint(0,2)}.{random.randint(1,20)}",
            "src_port":    random.randint(1024, 40000),
            "dst_port":    random.choice([80, 443, 53, 8080, 8443]),
            "protocol":    random.choice(protocols),
            "packet_size": random.randint(64, 900),
            "ttl":         random.randint(55, 128),
            "flags":       random.choice(["SA", "A", "PA", ""]),
        }


def get_feature_vector(packet_data: dict) -> list:
    """Returns a list of feature values in fixed order for ML model."""
    f = extract_features_from_packet(packet_data)
    return [f[col] for col in FEATURE_COLUMNS]


if __name__ == "__main__":
    pkt = simulate_packet(is_attack=False)
    print("Simulated Normal Packet:", pkt)
    print("Features:", extract_features_from_packet(pkt))

    pkt_atk = simulate_packet(is_attack=True)
    print("\nSimulated Attack Packet:", pkt_atk)
    print("Features:", extract_features_from_packet(pkt_atk))