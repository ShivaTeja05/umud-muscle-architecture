"""Extract only human validation PA/FL from public GasFasTrack MATLAB files.

Input is the gasfas_mat_files.zip exported by our CPU schema audit. Automated
tracks are never substituted for missing validation labels. Output remains an
unpaired label inventory until corresponding raw images have been verified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def validations(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == 'validate':
                yield path + (key,), value
            elif not key.startswith('__'):
                yield from validations(value, path + (key,))


def records(value, identity):
    if not isinstance(value, dict):
        raise ValueError('validate is not a MATLAB structure')
    arrays = {key: np.asarray(value[key], dtype=float).reshape(-1)
              for key in ('trackFrames', 'length', 'pennation')}
    n = len(arrays['trackFrames'])
    if not n or any(len(a) != n for a in arrays.values()):
        raise ValueError('validation frame/label counts disagree or are empty')
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('nonfinite validation frame or label')
    frames = arrays['trackFrames']
    if (frames < 1).any() or (frames != np.floor(frames)).any() or len(set(frames)) != n:
        raise ValueError('validation frame IDs must be unique positive MATLAB indices')
    if (arrays['length'] <= 0).any() or ((arrays['pennation'] <= 0) | (arrays['pennation'] >= 90)).any():
        raise ValueError('validation PA/FL outside expected physical bounds')
    for i in range(n):
        yield dict(identity, frame_matlab_1based=int(frames[i]),
                   pa_deg=float(arrays['pennation'][i]), fl_mm=float(arrays['length'][i]),
                   provenance='fascicle.validate; aponeuroses inherited from automatic tracking')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows, errors, no_labels = [], [], []
    pattern = re.compile(r'^validation_data/Day (\d+)/Examiner (\d+)/Subject (\d+)/TrackedData\.mat$')
    with zipfile.ZipFile(args.archive) as archive:
        for name in sorted(archive.namelist()):
            match = pattern.match(name)
            if not match:
                continue  # session subdirectories may duplicate the root MAT
            data = loadmat(io.BytesIO(archive.read(name)), simplify_cells=True)
            day, examiner, subject = match.groups()
            found = list(validations(data))
            if not found:
                no_labels.append(name)
            for path, value in found:
                identity = dict(subject=subject, day=day, examiner=examiner,
                                mat_path=name, struct_path='/'.join(path))
                try:
                    rows.extend(records(value, identity))
                except (ValueError, KeyError, TypeError) as exc:
                    errors.append(dict(identity, error=str(exc)))
    args.out.mkdir(parents=True, exist_ok=True)
    repeats = Counter((r['subject'], r['struct_path'], r['frame_matlab_1based']) for r in rows)
    # Broad review flags, not automatic exclusions or fitted training cutoffs.
    # Near-parallel line intersections can yield unbounded extrapolated FL.
    review = [r for r in rows if r['fl_mm'] > 300 or r['pa_deg'] < 3]
    report = dict(source='https://zenodo.org/records/2598553',
                  archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),
                  n_labels=len(rows), subjects=sorted({r['subject'] for r in rows}),
                  n_named_frames=len(repeats),
                  ratings_per_named_frame=dict(Counter(repeats.values())),
                  n_geometry_review_records=len(review),
                  geometry_review_records=review,
                  errors=errors, root_files_without_validation=no_labels,
                  ready_for_training=False,
                  reason='Raw image correspondence, overlap checks and measurement conventions remain to be verified.')
    (args.out / 'inventory.json').write_text(json.dumps(report, indent=2) + '\n')
    if errors:
        raise RuntimeError(f'{len(errors)} invalid validation structures; see inventory.json')
    if not rows:
        raise RuntimeError('no manual validation labels found; see inventory.json')
    with (args.out / 'manual_validation_labels.csv').open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
