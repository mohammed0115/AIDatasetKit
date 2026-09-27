# final_L

## environment
```
label=final_L
commit=fd4cdd2eddc561866e5e43b852fdcfffda9eb982
python=python
python_version 3.12.3
platform Windows-11-10.0.26200-SP0
locale cp1252
import_path <scratch>\trees\final_L\aidatasetkit\__init__.py
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
4097 tests collected in 68.14s (0:01:08)
```

## result
```
FAILED tests/integration/test_visualization_without_matplotlib.py::TestPlanningWithoutMatplotlib::test_a_full_plan_is_produced_and_serialised
1 failed, 4051 passed, 46 skipped in 6911.63s (1:55:11)
exit=1 seconds=6930
```
