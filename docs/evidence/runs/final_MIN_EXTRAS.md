# final_MIN_EXTRAS

## environment
```
label=final_MIN_EXTRAS
commit=fd4cdd2eddc561866e5e43b852fdcfffda9eb982
python=<scratch>/venv_min_extras/Scripts/python.exe
python_version 3.11.16
platform Windows-10-10.0.26200-SP0
locale cp1252
import_path <scratch>\trees\final_MIN_EXTRAS\aidatasetkit\__init__.py
catboost==1.2
lightgbm==4.0.0
matplotlib==3.9.0
numpy==1.26.4
pandas==2.1.4
pytest==8.0.0
scikit-learn==1.6.1
scipy==1.11.4
xgboost==2.0.0
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
4119 tests collected in 20.28s
```

## result
```
4076 passed, 43 skipped, 1 warning in 412.61s (0:06:52)
exit=0 seconds=416
```
