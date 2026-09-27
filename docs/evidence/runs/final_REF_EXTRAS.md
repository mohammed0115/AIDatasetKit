# final_REF_EXTRAS

## environment
```
label=final_REF_EXTRAS
commit=fd4cdd2eddc561866e5e43b852fdcfffda9eb982
python=<scratch>/venv_extras/Scripts/python.exe
python_version 3.12.3
platform Windows-11-10.0.26200-SP0
locale cp1252
import_path <scratch>\trees\final_REF_EXTRAS\aidatasetkit\__init__.py
catboost==1.2.10
lightgbm==4.7.0
matplotlib==3.11.2
numpy==2.5.2
pandas==3.0.5
pytest==8.3.3
scikit-learn==1.9.0
scipy==1.18.0
xgboost==3.4.1
```

## command
```
git archive <commit> | tar -x -C <tree>; cd <tree>
<venv python> -m pip install --no-deps <tree>        # venvs only, for distribution metadata
<python> -m pytest -p no:cacheprovider --collect-only -q
<python> -m pytest -p no:cacheprovider -q -rfE --junitxml=junit.xml   # PYTHONDONTWRITEBYTECODE=1
```

## collect
```
4119 tests collected in 15.79s
```

## result
```
4076 passed, 43 skipped in 494.00s (0:08:14)
exit=0 seconds=497
```
