#!/usr/bin/env python3
"""Foreground authenticated Jupyter using an existing Sage image; never build/pull."""
import argparse
from pathlib import Path
import subprocess
import uuid
from notes import outside_repo

REPO = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', required=True, help='Existing Sage image ID or digest')
    p.add_argument('--port', type=int, default=8889)
    p.add_argument('--scratch', type=Path, default=REPO.parent.parent/'Data/ReadingRoom/propulsion/notebooks')
    a = p.parse_args()
    if not 1 <= a.port <= 65535:
        p.error('Port must be 1..65535')
    scratch = outside_repo(REPO,a.scratch)
    scratch.mkdir(parents=True,exist_ok=True)
    image = subprocess.check_output(['docker','image','inspect',a.image,'--format','{{.Id}}'],text=True).strip()
    # The image's own user needs write access. Match the host UID/GID for new notebooks.
    import os
    name = 'propulsion-sage-'+uuid.uuid4().hex[:12]
    command = ['docker','run','--rm','--name',name,'--pull=never','--user',f'{os.getuid()}:{os.getgid()}',
               '-e','HOME=/tmp','--publish',f'127.0.0.1:{a.port}:8888',
               '--mount',f'type=bind,src={REPO / "Studies"},dst=/studies,readonly',
               '--mount',f'type=bind,src={scratch},dst=/work','--workdir','/work',image,
               'sage','-n','jupyter','--no-browser','--ip=0.0.0.0','--port=8888']
    print(f'Open the token URL from Jupyter using localhost port {a.port}. Ctrl+C stops it.',flush=True)
    try:
        subprocess.run(command,check=True)
    except KeyboardInterrupt:
        pass
    finally:
        subprocess.run(['docker','stop','--time','3',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)


if __name__ == '__main__':
    main()
