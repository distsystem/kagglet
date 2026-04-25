"""Real-time streaming logs for Kaggle kernels via Firebase SSE.

The kaggle CLI only offers polling `kernels_output`. This module taps the same
Firestore SSE stream Kaggle's browser UI uses, so cell output arrives within a
second of being printed.

Auth chain:  browser cookies → Kaggle internal API → Firebase custom token →
             Firestore LogsURL → SSE.

`cookies_from_chrome()` extracts cookies from Linux Chrome (via GNOME Keyring).
If you're on another OS or browser, pass a `cookies` dict directly.
"""

import os
import sys
import json
import time

import requests

KAGGLE_INTERNAL = "https://www.kaggle.com/api/i"
FIREBASE_SIGN_IN = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithCustomToken"
FIRESTORE_DOCS = "https://firestore.googleapis.com/v1/projects/{project}/databases/(default)/documents"

KAGGLE_COOKIE_NAMES = ("XSRF-TOKEN", "__Host-KAGGLEID", "ka_sessionid", "CSRF-TOKEN", "ka_db")

_CHROME_COOKIE_PATHS = [
    os.path.expanduser("~/.config/google-chrome-beta/Default/Cookies"),
    os.path.expanduser("~/.config/google-chrome/Default/Cookies"),
    os.path.expanduser("~/.config/chromium/Default/Cookies"),
]


def cookies_from_chrome() -> dict[str, str]:
    """Decrypt Kaggle cookies from Chrome on Linux (GNOME Keyring + AES-128-CBC)."""
    import shutil
    import sqlite3
    import tempfile
    import subprocess

    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers import Cipher, modes, algorithms
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    key_b64 = subprocess.run(
        ["secret-tool", "lookup", "application", "chrome"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    kdf = PBKDF2HMAC(algorithm=hashes.SHA1(), length=16, salt=b"saltysalt", iterations=1)
    aes_key = kdf.derive(key_b64.encode())

    db_path = next((p for p in _CHROME_COOKIE_PATHS if os.path.exists(p)), None)
    if not db_path:
        raise FileNotFoundError("Chrome cookies DB not found")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = os.path.join(tmpdir, "cookies.db")
        shutil.copy2(db_path, tmp)
        with sqlite3.connect(tmp) as conn:
            rows = conn.execute(
                f"SELECT name, encrypted_value FROM cookies "
                f"WHERE host_key LIKE '%kaggle.com' AND name IN ({','.join('?' * len(KAGGLE_COOKIE_NAMES))})",
                KAGGLE_COOKIE_NAMES,
            ).fetchall()

    cookies = {}
    for name, enc_val in rows:
        raw = enc_val[3:] if enc_val[:3] == b"v11" else enc_val
        dec = Cipher(algorithms.AES128(aes_key), modes.CBC(b" " * 16)).decryptor()
        pt = dec.update(raw) + dec.finalize()
        pt = pt[32:]  # skip Chrome padding block
        cookies[name] = pt[: -pt[-1]].decode()

    missing = set(KAGGLE_COOKIE_NAMES) - cookies.keys()
    if missing:
        raise RuntimeError(f"missing cookies: {missing}")
    return cookies


class KaggleSession:
    """HTTP session with Kaggle browser cookies for internal API calls."""

    def __init__(self, cookies: dict[str, str]):
        self._s = requests.Session()
        self._s.headers.update(
            {
                "Content-Type": "application/json",
                "Origin": "https://www.kaggle.com",
                "X-XSRF-TOKEN": cookies["XSRF-TOKEN"],
            }
        )
        self._s.cookies.update(cookies)

    def internal(self, service: str, method: str, body: dict | None = None) -> dict:
        r = self._s.post(f"{KAGGLE_INTERNAL}/{service}/{method}", json=body or {})
        r.raise_for_status()
        return r.json()


class FirebaseAuth:
    """Firebase auth chain: Kaggle cookies → custom token → Firebase ID token."""

    def __init__(self, session: KaggleSession):
        config = session.internal("kernels.KernelsService", "GetFirebaseConfig")
        self.project_id: str = config["projectId"]
        self._api_key: str = config["apiKey"]
        self._session = session
        self._id_token: str = ""
        self._refresh_token: str = ""
        self._expires_at: float = 0
        self._authenticate()

    def _set_tokens(self, id_token: str, refresh_token: str, expires_in) -> None:
        self._id_token = id_token
        self._refresh_token = refresh_token
        self._expires_at = time.monotonic() + int(expires_in or 3600) - 60

    def _authenticate(self):
        custom = self._session.internal("kernels.KernelsService", "GetFirebaseAuthToken")
        r = requests.post(
            f"{FIREBASE_SIGN_IN}?key={self._api_key}",
            json={"token": custom["authToken"], "returnSecureToken": True},
        )
        r.raise_for_status()
        data = r.json()
        self._set_tokens(data["idToken"], data["refreshToken"], data.get("expiresIn"))

    def _refresh(self):
        r = requests.post(
            f"https://securetoken.googleapis.com/v1/token?key={self._api_key}",
            json={"grant_type": "refresh_token", "refresh_token": self._refresh_token},
        )
        r.raise_for_status()
        data = r.json()
        self._set_tokens(data["id_token"], data["refresh_token"], data.get("expires_in"))

    @property
    def id_token(self) -> str:
        if time.monotonic() >= self._expires_at:
            self._refresh()
        return self._id_token


def find_kernel_run_id(session: KaggleSession, slug: str) -> int:
    """Get kernelRunId for the most recent session matching a kernel slug."""
    _, kernel_slug = slug.split("/")

    data = session.internal("kernels.KernelsService", "ListKernelSessions")
    for s in data.get("sessions", []):
        title = s.get("title", "")
        run_id = s.get("kernelRunId")
        # Title comparison is fuzzy — Kaggle displays the human title, not the slug.
        expected = kernel_slug.replace("-", " ")
        if expected.lower() in title.lower() and run_id:
            return run_id

    raise RuntimeError(f"no active session found for {slug}")


def get_sse_url(session: KaggleSession, auth: FirebaseAuth, kernel_run_id: int) -> str:
    """Register Firestore session and fetch the SSE logs URL."""
    resp = session.internal(
        "kernels.KernelsService",
        "UpdateUserKernelFirestoreAuth",
        {"firebaseIdToken": auth.id_token, "kernelRunId": kernel_run_id},
    )
    session_id = resp["sessionId"]

    doc_url = FIRESTORE_DOCS.format(project=auth.project_id)
    r = requests.get(
        f"{doc_url}/sessions/{session_id}/data/LogsURL",
        headers={"Authorization": f"Bearer {auth.id_token}"},
    )
    r.raise_for_status()
    return r.json()["fields"]["url"]["stringValue"]


def consume_sse(url: str, timeout: int):
    """Connect to SSE endpoint and print log events."""
    with requests.get(url, stream=True, headers={"Accept": "text/event-stream"}, timeout=timeout) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "END_OF_LOG":
                break
            try:
                entry = json.loads(payload)
            except json.JSONDecodeError:
                continue
            stream = entry.get("stream_name", "stdout")
            data = entry.get("data", "")
            if stream == "stderr":
                sys.stderr.write(f"\033[2m{data}\033[0m")
            else:
                sys.stdout.write(data)
            sys.stdout.flush()


def stream_logs(
    slug: str,
    *,
    cookies: dict[str, str] | None = None,
    timeout: int = 7200,
    session_wait: int = 120,
    sse_wait: int = 180,
) -> None:
    """Stream real-time kernel logs to stdout/stderr.

    Args:
        slug: kernel slug in `{owner}/{kernel}` form.
        cookies: Kaggle browser cookies. Defaults to `cookies_from_chrome()`.
        timeout: max seconds to keep the SSE connection open.
        session_wait: max seconds to wait for the kernel session to appear.
        sse_wait: max seconds to wait for the Firestore LogsURL to appear.
    """
    if cookies is None:
        cookies = cookies_from_chrome()
    session = KaggleSession(cookies)
    auth = FirebaseAuth(session)

    t0 = time.monotonic()
    kernel_run_id = None
    while time.monotonic() - t0 < session_wait:
        try:
            kernel_run_id = find_kernel_run_id(session, slug)
            break
        except RuntimeError:
            time.sleep(5)
    if kernel_run_id is None:
        raise RuntimeError(f"no running session found for {slug}")

    print(f"\033[2m[streaming {slug} run={kernel_run_id}]\033[0m", file=sys.stderr)

    sse_url = None
    while time.monotonic() - t0 < sse_wait:
        try:
            sse_url = get_sse_url(session, auth, kernel_run_id)
            break
        except (KeyError, requests.HTTPError):
            time.sleep(5)
    if sse_url is None:
        raise RuntimeError("SSE URL not available in Firestore")

    consume_sse(sse_url, timeout)
