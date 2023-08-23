## Developing

```
pip install -e .
```

## Code cleanup

Installing code management tools:
```
sudo apt install isort
pip install black
```

Running: 
```
isort .
python -m black .
```


## Package Management

### UPDATE the version in `setup.py`

Uploading to PyPI

```
python -m pip install --upgrade setuptools twine wheel build
python -m build
python -m twine upload dist/*

```
