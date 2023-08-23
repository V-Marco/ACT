## Developing

```
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
black .
```


## Package Management

### UPDATE the version in `setup.py`

Uploading to PyPI
```python
# Install setuptools and wheel
python -m pip install --user --upgrade setuptools wheel

# Run from setup.py directory
python setup.py sdist bdist_wheel

# Files will be generated in the dist directory
dist/
  example_pkg_your_username-0.0.1-py3-none-any.whl
  example_pkg_your_username-0.0.1.tar.gz
  
# Install Twine
python -m pip install --user --upgrade twine

# Upload to Test
python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# Upload to PyPI
python -m twine upload dist/*

# Install from Test
python -m pip install --index-url https://test.pypi.org/simple/ example-pkg-your-username

# Install from PyPI
python -m pip install example-pkg-your-username

```

## All together
```
python -m pip install --upgrade setuptools twine wheel 
python setup.py sdist bdist_wheel
python -m twine upload dist/*

```
