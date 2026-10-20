"""Upload the files listed in ``tools/drive_upload_manifest.yaml`` to Google Drive.

Usage (from the repository root):

    python tools/upload_to_drive.py --dry-run          # list what would be uploaded, change nothing
    python tools/upload_to_drive.py                    # upload (backend chosen automatically)
    python tools/upload_to_drive.py --backend api      # force the Google Drive API backend

Backends
--------
``local`` (preferred, no setup): copies into the "My Drive" folder of Google Drive for Desktop, which then syncs. The
folder is found automatically (``G:\\My Drive`` on Windows, ``~/Library/CloudStorage/GoogleDrive-*/My Drive`` on macOS, …)
or given with ``--drive-dir`` / the ``NFLBOWL_DRIVE_DIR`` environment variable.

``api`` (when Drive for Desktop is not installed): Google Drive API v3 with a one-time browser login.
    1. pip install google-api-python-client google-auth-oauthlib      (or: pip install -e .[drive])
    2. Google Cloud console → APIs & Services → enable "Google Drive API" → Credentials → Create OAuth client ID →
       "Desktop app" → download the JSON as ``tools/drive_credentials.json`` (git-ignored).
    3. Run the script: a browser window asks for consent once; the token is cached in ``~/.nflbowl/drive_token.json``.
    Scope ``drive`` (full) is needed to find the EXISTING ``NFLBowl`` folder the notebooks use; with the narrower
    ``drive.file`` scope the script could only see folders it created itself and would make a duplicate ``NFLBowl``.

Behaviour (both backends): folders are created when missing; a file whose MD5 equals the Drive copy is skipped; a changed
file is replaced in place (same Drive file id, so links and Colab "Open with" stay valid); nothing is ever deleted.
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import hashlib
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO / "tools" / "drive_upload_manifest.yaml"
DEFAULT_CREDENTIALS = REPO / "tools" / "drive_credentials.json"
TOKEN = Path.home() / ".nflbowl" / "drive_token.json"
SCOPES = ["https://www.googleapis.com/auth/drive"]
FOLDER_MIME = "application/vnd.google-apps.folder"


# ---------------------------------------------------------------------- manifest
def load_manifest(path: Path) -> dict:
    import yaml

    m = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not m.get("drive_root") or not m.get("entries"):
        raise SystemExit(f"{path}: needs 'drive_root' and a non-empty 'entries' list")
    return m


def plan(manifest: dict) -> list[tuple[Path, str]]:
    """[(local file, destination folder relative to drive_root)] — sorted, de-duplicated, exclusions applied."""
    excl = [str(e) for e in manifest.get("exclude") or []]
    out: dict[Path, str] = {}
    for e in manifest["entries"]:
        hits = sorted(Path(p) for p in glob.glob(str(REPO / e["source"]), recursive=True))
        files = [p for p in hits if p.is_file()]
        if not files:
            print(f"warning: no file matches {e['source']!r}")
        for p in files:
            rel = p.relative_to(REPO).as_posix()
            if any(fnmatch.fnmatch(rel, x) for x in excl):
                continue
            out[p] = str(e.get("dest", "")).strip("/")
    return sorted(out.items())


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------- local backend (Google Drive for Desktop)
def find_drive_dir() -> Path | None:
    env = os.environ.get("NFLBOWL_DRIVE_DIR")
    if env:
        return Path(env)
    cands: list[Path] = []
    if os.name == "nt":
        for letter in "GHIJKLMNOPQRSTUVWXYZDEF":
            cands += [Path(f"{letter}:/My Drive"), Path(f"{letter}:/Mon Drive"), Path(f"{letter}:/Mi unidad")]
    home = Path.home()
    cands += sorted(home.glob("Library/CloudStorage/GoogleDrive-*/My Drive")) + [home / "Google Drive" / "My Drive", home / "Google Drive",
                                                                               Path("/Volumes/GoogleDrive/My Drive")]
    return next((c for c in cands if c.is_dir()), None)


def upload_local(items: list[tuple[Path, str]], drive_dir: Path, root: str, dry: bool) -> tuple[int, int]:
    done = skipped = 0
    for src, dest in items:
        target = drive_dir / root / dest / src.name
        if target.is_file() and md5(target) == md5(src):
            skipped += 1
            continue
        print(f"{'would copy' if dry else 'copy'}  {src.relative_to(REPO)}  →  {target}")
        if not dry:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f".{target.name}.uploading")
            shutil.copyfile(src, tmp)
            os.replace(tmp, target)
        done += 1
    return done, skipped


# ---------------------------------------------------------------------- API backend
def drive_service(credentials: Path):
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        raise SystemExit("the api backend needs: pip install google-api-python-client google-auth-oauthlib")
    creds = Credentials.from_authorized_user_file(str(TOKEN), SCOPES) if TOKEN.is_file() else None
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    if not creds or not creds.valid:
        if not credentials.is_file():
            raise SystemExit(f"OAuth client file not found: {credentials} (see the module docstring, step 2)")
        creds = InstalledAppFlow.from_client_secrets_file(str(credentials), SCOPES).run_local_server(port=0)
        TOKEN.parent.mkdir(parents=True, exist_ok=True)
        TOKEN.write_text(creds.to_json(), encoding="utf-8")
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _q(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


def _child(svc, parent: str, name: str, folder: bool) -> dict | None:
    mime = f"mimeType = '{FOLDER_MIME}'" if folder else f"mimeType != '{FOLDER_MIME}'"
    r = svc.files().list(q=f"'{parent}' in parents and name = '{_q(name)}' and {mime} and trashed = false",
                         fields="files(id, name, md5Checksum, modifiedTime)", spaces="drive", pageSize=10).execute()
    files = r.get("files", [])
    if len(files) > 1:
        print(f"warning: {len(files)} items named {name!r} in the same Drive folder; using the most recently modified")
        files.sort(key=lambda f: f.get("modifiedTime", ""), reverse=True)
    return files[0] if files else None


def ensure_folder(svc, path: str, cache: dict, dry: bool) -> str:
    """Folder id of ``My Drive/<path>``, creating missing parts (in a dry run a placeholder id is returned)."""
    parent = "root"
    walked = ""
    for part in [p for p in path.split("/") if p]:
        walked = f"{walked}/{part}"
        if walked in cache:
            parent = cache[walked]
            continue
        found = _child(svc, parent, part, folder=True) if not parent.startswith("dry:") else None
        if found:
            parent = found["id"]
        elif dry:
            print(f"would create folder  My Drive{walked}")
            parent = f"dry:{walked}"
        else:
            parent = svc.files().create(body={"name": part, "mimeType": FOLDER_MIME, "parents": [parent]}, fields="id").execute()["id"]
            print(f"created folder  My Drive{walked}")
        cache[walked] = parent
    return parent


def upload_api(items: list[tuple[Path, str]], root: str, credentials: Path, dry: bool) -> tuple[int, int]:
    from googleapiclient.http import MediaFileUpload

    svc = drive_service(credentials)
    cache: dict[str, str] = {}
    done = skipped = 0
    for src, dest in items:
        folder = ensure_folder(svc, f"{root}/{dest}", cache, dry)
        existing = _child(svc, folder, src.name, folder=False) if not folder.startswith("dry:") else None
        if existing and existing.get("md5Checksum") == md5(src):
            skipped += 1
            continue
        action = "update" if existing else "upload"
        print(f"{'would ' + action if dry else action}  {src.relative_to(REPO)}  →  My Drive/{root}/{dest}/{src.name}")
        if not dry:
            media = MediaFileUpload(str(src), resumable=src.stat().st_size > 5 * 2 ** 20)
            if existing:
                svc.files().update(fileId=existing["id"], media_body=media).execute()
            else:
                svc.files().create(body={"name": src.name, "parents": [folder]}, media_body=media, fields="id").execute()
        done += 1
    return done, skipped


# ---------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--backend", choices=("auto", "local", "api"), default="auto")
    ap.add_argument("--drive-dir", type=Path, default=None, help="Google Drive for Desktop 'My Drive' folder (local backend)")
    ap.add_argument("--credentials", type=Path, default=DEFAULT_CREDENTIALS, help="OAuth client JSON (api backend)")
    ap.add_argument("--dry-run", action="store_true", help="show what would change, change nothing")
    a = ap.parse_args(argv)

    m = load_manifest(a.manifest)
    items = plan(m)
    if not items:
        print("nothing to upload")
        return 1
    print(f"{len(items)} file(s) in the manifest → My Drive/{m['drive_root']}")
    drive_dir = a.drive_dir or (find_drive_dir() if a.backend in ("auto", "local") else None)
    if a.backend == "local" or (a.backend == "auto" and drive_dir):
        if not drive_dir or not drive_dir.is_dir():
            raise SystemExit("Google Drive for Desktop folder not found: pass --drive-dir or set NFLBOWL_DRIVE_DIR (or use --backend api)")
        print(f"backend local: {drive_dir}")
        done, skipped = upload_local(items, drive_dir, m["drive_root"], a.dry_run)
    else:
        print("backend api: Google Drive API v3")
        done, skipped = upload_api(items, m["drive_root"], a.credentials, a.dry_run)
    print(f"{'would change' if a.dry_run else 'changed'} {done} file(s); {skipped} unchanged (skipped)")
    if a.backend != "api" and drive_dir and not a.dry_run:
        print("Drive for Desktop syncs in the background: wait for it to finish before opening the files in Colab.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
