# final_MIN

## environment
```
label=final_MIN
commit=fd4cdd2eddc561866e5e43b852fdcfffda9eb982
python=<scratch>/venv_min/Scripts/python.exe
python_version 3.11.16
platform Windows-10-10.0.26200-SP0
locale cp1252
import_path <scratch>\trees\final_MIN\aidatasetkit\__init__.py
numpy==1.26.4
pandas==2.1.4
pytest==8.0.0
scikit-learn==1.6.1
scipy==1.11.4
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
4097 tests collected in 101.10s (0:01:41)
```

## result
```
4052 passed, 46 skipped in 6693.21s (1:51:33)
exit=0 seconds=6702
```
