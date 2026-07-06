"""
Echo-Vision - Orin Nano side of the LAN bridge (module 2).

Run on the Orin Nano:
    python3 client.py <pi5-ip-address>

What it does, each poll cycle (default every 2s):
  1. GET /api/state from the Pi 5.
  2. For anyone newly IDENTIFIED, announce them (with a cooldown so it
     doesn't repeat "X is nearby" every single poll).
  3. For anyone PENDING (unidentified), ask the user yes/no + name and
     POST the decision back via /api/register or /api/decline.
  4. If the Pi 5 is unreachable, says so and keeps retrying -- it never
     crashes just because the LAN cable or the Pi 5 process is briefly down.

INTEGRATION NOTE for module 3 (voice interface):
  Exactly like the Pi 5's app.py, the three functions below (`announce`,
  `ask_yes_no`, `ask_name`) are the only things that need to change once
  the mic + local LLM exist. Everything in the Bridge class that talks to
  the Pi 5 over HTTP stays the same.
"""

import sys
import time

import requests


def announce(text: str):
    """Placeholder for TTS output via the local LLM. Swapped in module 3."""
    print(f"[ORIN] {text}")


def ask_yes_no(prompt: str) -> bool:
    """Placeholder for mic + STT. Swapped in module 3."""
    return input(f"{prompt} (y/n): ").strip().lower().startswith("y")


def ask_name() -> str:
    """Placeholder for mic + STT. Swapped in module 3."""
    return input("What is this person's name? ").strip()


class EchoVisionClient:
    """Thin HTTP wrapper around the Pi 5's face-ID API."""

    def __init__(self, host: str, port: int = 5000, timeout: float = 5.0):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout

    def get_state(self) -> dict:
        r = requests.get(f"{self.base}/api/state", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def register(self, pending_id: str, name: str) -> dict:
        r = requests.post(
            f"{self.base}/api/register",
            json={"pending_id": pending_id, "name": name},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def decline(self, pending_id: str) -> dict:
        r = requests.post(
            f"{self.base}/api/decline",
            json={"pending_id": pending_id},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()


class Bridge:
    """
    Polling loop + de-duplication logic, kept separate from EchoVisionClient
    so it can be unit-tested with a fake client and injected answer functions.
    """

    def __init__(self, client: EchoVisionClient, poll_interval: float = 2.0,
                 identified_cooldown: float = 15.0):
        self.client = client
        self.poll_interval = poll_interval
        self.identified_cooldown = identified_cooldown
        self._last_identified_announce: dict[str, float] = {}
        self._handled_pending: set[str] = set()

    def poll_once(self):
        try:
            state = self.client.get_state()
        except requests.RequestException as e:
            announce(f"Pi 5 is unreachable ({e}); will keep retrying.")
            return

        now = time.time()

        for person in state.get("identified", []):
            name = person["name"]
            last = self._last_identified_announce.get(name, 0)
            if now - last > self.identified_cooldown:
                announce(f"{name} is nearby.")
                self._last_identified_announce[name] = now

        # Anyone no longer pending (answered elsewhere, e.g. the browser UI)
        # can be forgotten so the de-dup set doesn't grow forever.
        still_pending_ids = {ev["id"] for ev in state.get("pending", [])}
        self._handled_pending &= still_pending_ids

        for ev in state.get("pending", []):
            pid = ev["id"]
            if pid in self._handled_pending:
                continue
            self._handled_pending.add(pid)

            announce("Unidentified person detected.")
            if ask_yes_no("Should I add this person to the list?"):
                name = ask_name()
                if name:
                    self.client.register(pid, name)
                    announce(f"Added {name}.")
                else:
                    self.client.decline(pid)
                    announce("No name given, skipping.")
            else:
                self.client.decline(pid)
                announce("Okay, not adding them.")

    def run_forever(self):
        announce("Connected to the Echo-Vision Pi 5 bridge.")
        while True:
            self.poll_once()
            time.sleep(self.poll_interval)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 client.py <pi5-ip-address> [port]")
        sys.exit(1)
    host = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    client = EchoVisionClient(host, port)
    Bridge(client).run_forever()


if __name__ == "__main__":
    main()