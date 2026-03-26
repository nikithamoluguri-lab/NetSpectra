"""
Module 4: Network Traffic Capture Module
Captures live packets OR simulates traffic for testing.
Uses Scapy if available; falls back to simulation automatically.
"""

import time
import random
import threading
from datetime import datetime
from feature_extractor import simulate_packet

# Try importing Scapy (requires root / admin on most OS)
try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("[Capture] Scapy not available — using simulation mode.")


# ──────────────────────────────────────────
# Live capture (Scapy)
# ──────────────────────────────────────────

def _parse_scapy_packet(packet) -> dict | None:
    """Convert a Scapy packet to our standard dict format."""
    if not packet.haslayer(IP):
        return None

    ip_layer  = packet[IP]
    protocol  = "OTHER"
    src_port  = 0
    dst_port  = 0
    flags     = ""

    if packet.haslayer(TCP):
        tcp = packet[TCP]
        protocol = "TCP"
        src_port = tcp.sport
        dst_port = tcp.dport
        flags    = str(tcp.flags)
    elif packet.haslayer(UDP):
        udp = packet[UDP]
        protocol = "UDP"
        src_port = udp.sport
        dst_port = udp.dport
    elif packet.haslayer(ICMP):
        protocol = "ICMP"

    return {
        "timestamp":   datetime.now().isoformat(),
        "src_ip":      ip_layer.src,
        "dst_ip":      ip_layer.dst,
        "src_port":    src_port,
        "dst_port":    dst_port,
        "protocol":    protocol,
        "packet_size": len(packet),
        "ttl":         ip_layer.ttl,
        "flags":       flags,
    }


def start_live_capture(callback, interface=None, packet_count=0):
    """
    Start live capture using Scapy.
    callback(packet_dict) is called for each parsed packet.
    packet_count=0 means capture indefinitely.
    NOTE: Requires root / admin privileges.
    """
    if not SCAPY_AVAILABLE:
        raise RuntimeError("Scapy is not installed.")

    def _handler(pkt):
        data = _parse_scapy_packet(pkt)
        if data:
            callback(data)

    kwargs = {"prn": _handler, "store": False}
    if interface:
        kwargs["iface"] = interface
    if packet_count:
        kwargs["count"] = packet_count

    print(f"[Capture] Starting live capture on interface: {interface or 'default'}")
    sniff(**kwargs)


# ──────────────────────────────────────────
# Simulation mode (no Scapy / no root needed)
# ──────────────────────────────────────────

_simulation_running = False
_simulation_thread  = None


def start_simulation(callback, packets_per_second: float = 2.0, attack_ratio: float = 0.08):
    """
    Simulate a stream of packets and call callback(packet_dict) for each.
    Runs in a background daemon thread.
    """
    global _simulation_running, _simulation_thread

    if _simulation_running:
        print("[Capture] Simulation already running.")
        return

    _simulation_running = True
    delay = 1.0 / max(packets_per_second, 0.1)

    def _loop():
        global _simulation_running
        print(f"[Capture] Simulation started — {packets_per_second} pkt/s, attack ratio={attack_ratio}")
        while _simulation_running:
            is_attack = random.random() < attack_ratio
            pkt = simulate_packet(is_attack=is_attack)
            try:
                callback(pkt)
            except Exception as e:
                print(f"[Capture] Callback error: {e}")
            time.sleep(delay)
        print("[Capture] Simulation stopped.")

    _simulation_thread = threading.Thread(target=_loop, daemon=True)
    _simulation_thread.start()


def stop_simulation():
    global _simulation_running
    _simulation_running = False


def is_simulation_running() -> bool:
    return _simulation_running


# ──────────────────────────────────────────
# Auto-mode: live if available, else simulate
# ──────────────────────────────────────────

def auto_start(callback, **kwargs):
    """Starts live capture if Scapy + root available, otherwise simulation."""
    if SCAPY_AVAILABLE:
        try:
            t = threading.Thread(
                target=start_live_capture,
                args=(callback,),
                kwargs={"interface": kwargs.get("interface"), "packet_count": 0},
                daemon=True,
            )
            t.start()
            print("[Capture] Live capture thread started.")
        except Exception as e:
            print(f"[Capture] Live capture failed ({e}). Falling back to simulation.")
            start_simulation(callback, **kwargs)
    else:
        start_simulation(callback, **kwargs)


if __name__ == "__main__":
    import json

    def print_packet(pkt):
        print(json.dumps(pkt, indent=2))

    print("[Capture] Running 10-second simulation test …")
    start_simulation(print_packet, packets_per_second=1.0)
    time.sleep(10)
    stop_simulation()