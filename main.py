"""Recognize real 8x8 handwritten digits with train-only model selection."""
from resources import ensure_assets
ensure_assets()

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay

BASE=Path(__file__).resolve().parent
OUT=BASE/'outputs'


def main():
    OUT.mkdir(exist_ok=True)
    dataset=load_digits(as_frame=True)
    X,y=dataset.data,dataset.target
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
    assert not set(X_train.index)&set(X_test.index)
    pipeline=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(max_iter=3000,random_state=42))])
    search=GridSearchCV(pipeline,{'model__C':[.01,.1,1.,10.]},
                        cv=StratifiedKFold(5,shuffle=True,random_state=42),scoring='f1_macro',n_jobs=1)
    search.fit(X_train,y_train)
    pred=search.predict(X_test)
    baseline=DummyClassifier(strategy='most_frequent').fit(X_train,y_train).predict(X_test)
    scores={'dataset_rows':len(X),'features':X.shape[1],'classes':10,'train_rows':len(X_train),'test_rows':len(X_test),
            'best_C':search.best_params_['model__C'],'train_cv_macro_f1':float(search.best_score_),
            'test_accuracy':float(accuracy_score(y_test,pred)),'test_macro_f1':float(f1_score(y_test,pred,average='macro')),
            'baseline_accuracy':float(accuracy_score(y_test,baseline)),
            'baseline_macro_f1':float(f1_score(y_test,baseline,average='macro'))}
    (OUT/'metrics.json').write_text(json.dumps(scores,indent=2))
    (OUT/'classification_report.txt').write_text(classification_report(y_test,pred,zero_division=0))
    pd.DataFrame(search.cv_results_)[['param_model__C','mean_test_score','std_test_score','rank_test_score']].to_csv(OUT/'cv_results.csv',index=False)
    predictions=pd.DataFrame({'row_id':X_test.index,'actual_class':y_test.to_numpy(),'predicted_class':pred})
    predictions.to_csv(OUT/'test_predictions.csv',index=False)
    predictions.loc[predictions.actual_class!=predictions.predicted_class].to_csv(OUT/'error_analysis.csv',index=False)
    pd.DataFrame({'row_id':X_train.index,'split':'train'}).to_csv(OUT/'train_rows.csv',index=False)
    X_test.head(5).to_csv(OUT/'sample_input.csv',index=False)
    joblib.dump({'pipeline':search.best_estimator_,'features':list(X.columns)},OUT/'model.joblib')
    fig,ax=plt.subplots(figsize=(8,7))
    ConfusionMatrixDisplay.from_predictions(y_test,pred,ax=ax,colorbar=False,cmap='Blues')
    ax.set_title(f'Handwritten digits | held-out accuracy {scores["test_accuracy"]:.1%}')
    fig.tight_layout();fig.savefig(OUT/'evaluation.png',dpi=150);plt.close(fig)
    # Display real held-out examples, prioritizing errors rather than hiding them.
    incorrect=np.flatnonzero(pred!=y_test.to_numpy())
    positions=list(incorrect[:12])
    positions+=list(np.flatnonzero(pred==y_test.to_numpy())[:12-len(positions)])
    fig,axes=plt.subplots(2,6,figsize=(12,4.7))
    for ax,pos in zip(axes.flat,positions):
        ax.imshow(X_test.iloc[pos].to_numpy().reshape(8,8),cmap='gray_r',vmin=0,vmax=16)
        actual=int(y_test.iloc[pos]);guess=int(pred[pos])
        ax.set_title(f'True {actual} / predicted {guess}',fontsize=10,color='#c2413a' if actual!=guess else '#176b87')
        ax.axis('off')
    fig.suptitle('Real test examples: errors shown first, then correct examples',fontsize=13)
    fig.tight_layout();fig.savefig(OUT/'digit_examples.png',dpi=150);plt.close(fig)
    (OUT/'report.md').write_text(f'''# Handwritten digit results

- Dataset: {len(X):,} real digit images, 64 pixel features, ten classes (0-9).
- Train/test: {len(X_train)}/{len(X_test)}, stratified split, seed 42.
- Chosen C: {scores['best_C']}; training five-fold CV macro F1: {scores['train_cv_macro_f1']:.3f}.
- Held-out accuracy: {scores['test_accuracy']:.3f}; macro F1: {scores['test_macro_f1']:.3f}.
- Majority baseline accuracy: {scores['baseline_accuracy']:.3f}; macro F1: {scores['baseline_macro_f1']:.3f}.

This is a classic small benchmark of 8x8 grayscale digits, not a general OCR system.
The random split does not guarantee different writers in training and test sets, and performance
on photos, full documents, Arabic-script digits or a new writer is unverified. No such coverage
is claimed. Features remain inside a training-only scaling pipeline. Model selection uses
training CV; the held-out split is used for final evaluation only. Saved model uses training
rows only, matching these metrics. Error examples are intentionally shown, not hidden.

![Confusion matrix](evaluation.png)
![Held-out image examples](digit_examples.png)
''')
    (BASE/'data'/'dataset_description.txt').write_text(dataset.DESCR)
    print(json.dumps(scores,indent=2))


if __name__=='__main__':
    main()
