"""Prepare canonical arrays/splits from separately downloaded official raw data."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
NEW=ROOT/'experiments/imu_continuation_20261106'
OLD=ROOT/'experiments/imu_reproduction_20261028'


def main():
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',choices=['uci_har','pamap2','usc_had','realworld_har'],default=['uci_har','pamap2','usc_had','realworld_har'])
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    for name in a.datasets:
        script=NEW/('prepare.py' if name in ['uci_har','pamap2'] else 'prepare_extra.py')
        cmd=[sys.executable,str(script),'--datasets' if name in ['uci_har','pamap2'] else '--dataset',name]
        if a.dry_run:
            print(' '.join(cmd));continue
        subprocess.run(cmd,cwd=ROOT,check=True)
    if not a.dry_run:
        # Legacy models reuse the exact same UCI/PAMAP arrays and split files.
        for folder in ['data','manifests']:
            base=OLD/folder;base.mkdir(parents=True,exist_ok=True)
            if folder=='data':
                for name in ['uci_har','pamap2']:
                    path=base/name
                    if not path.exists():path.symlink_to(Path('../../imu_continuation_20261106/data')/name,target_is_directory=True)
            else:
                path=base/'splits'
                if not path.exists():path.symlink_to(Path('../../imu_continuation_20261106/manifests/splits'),target_is_directory=True)


if __name__=='__main__':main()
