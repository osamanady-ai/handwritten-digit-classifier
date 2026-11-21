"""Predict from a CSV of original feature columns after running main.py."""
from pathlib import Path
import argparse
import joblib
import pandas as pd


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True,help='CSV with the documented feature columns')
    args=parser.parse_args()
    model_path=Path(__file__).parent/'outputs'/'model.joblib'
    if not model_path.exists():
        parser.error('Run python main.py first to train the model.')
    # Never load untrusted downloaded joblib files: pickle-based formats execute code.
    bundle=joblib.load(model_path)
    frame=pd.read_csv(args.input)
    missing=set(bundle['features'])-set(frame.columns)
    if missing:
        parser.error('Missing feature columns: '+', '.join(sorted(missing)))
    features=frame[bundle['features']].apply(pd.to_numeric,errors='raise')
    if features.isna().any().any():
        parser.error('Input contains missing feature values.')
    if not features.ge(0).all().all() or not features.le(16).all().all():
        parser.error('Digit pixel intensities must be between 0 and 16.')
    frame['prediction']=bundle['pipeline'].predict(features)
    print(frame.to_csv(index=False),end='')


if __name__=='__main__':
    main()
