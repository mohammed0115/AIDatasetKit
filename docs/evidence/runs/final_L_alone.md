# final_L_alone

## environment
```
label=final_L_alone
commit=fd4cdd2eddc561866e5e43b852fdcfffda9eb982
python=python
python_version 3.12.3
platform Windows-11-10.0.26200-SP0
locale cp1252
import_path <scratch>\trees\final_L_alone\aidatasetkit\__init__.py
numpy==2.4.1
pandas==2.3.3
pytest==8.3.3
scikit-learn==1.8.0
scipy==1.17.0
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
4097 tests collected in 8.79s
```

## result
```
4052 passed, 46 skipped in 411.56s (0:06:51)
exit=0 seconds=414
```
