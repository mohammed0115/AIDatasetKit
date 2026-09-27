# final_REF

## environment
```
label=final_REF
commit=fd4cdd2eddc561866e5e43b852fdcfffda9eb982
python=<scratch>/venv_verified/Scripts/python.exe
python_version 3.12.3
platform Windows-11-10.0.26200-SP0
locale cp1252
import_path <scratch>\trees\final_REF\aidatasetkit\__init__.py
numpy==2.5.2
pandas==3.0.5
pytest==8.3.3
scikit-learn==1.9.0
scipy==1.18.0
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
4097 tests collected in 62.93s (0:01:02)
```

## result
```
4052 passed, 46 skipped in 6852.50s (1:54:12)
exit=0 seconds=6869
```
