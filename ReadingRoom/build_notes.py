#!/usr/bin/env python3
"""Build portrait and screen companions/master outside the repo, with provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from notes import input_hashes, outside_repo

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = outside_repo(REPO, args.output)
    output.mkdir(parents=True, exist_ok=True)
    inputs = input_hashes(REPO)
    products = {}
    with tempfile.TemporaryDirectory(prefix='propulsion-latex-') as directory:
        temp = Path(directory)
        for entry in ('master','numerical-recipes','wie','sutton'):
            source = 'master.tex' if entry == 'master' else 'companions/'+entry+'.tex'
            for edition in ('portrait','screen'):
                name = entry+'-'+edition
                expression = (r'\def\screenedition{1}' if edition == 'screen' else '') + r'\input{'+source+'}'
                command = ['pdflatex','-no-shell-escape','-halt-on-error','-interaction=nonstopmode',
                           '-output-directory',str(temp),'-jobname',name,expression]
                for _ in range(3):
                    run = subprocess.run(command,cwd=REPO/'documents/notes',text=True,capture_output=True)
                    if run.returncode:
                        raise RuntimeError(run.stdout[-6000:])
                    if _ == 0:
                        environment = dict(os.environ, BIBINPUTS=str(REPO/'documents/notes')+os.pathsep)
                        bibliography = subprocess.run(['bibtex',name],cwd=temp,env=environment,text=True,capture_output=True)
                        if bibliography.returncode:
                            raise RuntimeError(bibliography.stdout)
                log = (temp/(name+'.log')).read_text()
                if 'undefined' in log.lower() or 'multiply defined' in log.lower() or 'Overfull' in log:
                    raise RuntimeError('Unresolved reference or layout overflow in '+name+'\n'+log[-2500:])
                products[name+'.pdf'] = temp/(name+'.pdf')
        # Publish only after all eight builds pass. Provenance detects interrupted replacement.
        hashes = {}
        for filename,path in products.items():
            staging = output/('.'+filename+'.tmp')
            shutil.copyfile(path,staging)
            os.replace(staging,output/filename)
            hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        report = {'schema':1,'created':datetime.now(timezone.utc).isoformat(),'inputs':inputs,'outputs':hashes}
        with tempfile.NamedTemporaryFile(mode='w',dir=output,delete=False) as f:
            json.dump(report,f,indent=2)
            staging = f.name
        os.replace(staging,output/'build.json')
    print('Built eight PDFs with resolved citations and no overfull boxes: '+str(output))


if __name__ == '__main__':
    main()
