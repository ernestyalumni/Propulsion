#!/usr/bin/env python3
"""Record reproducible numerical/Sage evidence outside the source repository."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'ReadingRoom'))
from notes import input_hashes, outside_repo
from mechanics import numerical_checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sage-image', help='Existing prebuilt image ID/digest. Never pulled or built.')
    args = p.parse_args()
    output = outside_repo(REPO, args.output)
    numerical = numerical_checks()
    symbolic = None
    image = None
    if args.sage_image:
        image = subprocess.check_output(['docker','image','inspect',args.sage_image,'--format','{{.Id}}'], text=True).strip()
        command = ['docker','run','--rm','--pull=never','--network=none',
                   '--mount',f'type=bind,src={REPO / "Studies"},dst=/studies,readonly',
                   '--workdir','/tmp',image,'sage','-python','/studies/symbolic.py']
        run = subprocess.run(command, text=True, capture_output=True, timeout=180)
        if run.returncode:
            raise RuntimeError('Sage check failed; previous evidence was preserved.\n'+run.stderr+run.stdout)
        symbolic = json.loads(run.stdout)
    report = {'schema':1, 'created':datetime.now(timezone.utc).isoformat(),
              'python':sys.version.split()[0], 'image':image, 'inputs':input_hashes(REPO),
              'topics':{key:{'numerical':{'passed':True,'metrics':value},
                             'symbolic':symbolic['topics'][key] if symbolic else None}
                        for key,value in numerical.items()},
              'sage_version':symbolic['sage_version'] if symbolic else None}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=output.parent, delete=False) as f:
        json.dump(report,f,indent=2)
        temporary = f.name
    os.replace(temporary, output)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
