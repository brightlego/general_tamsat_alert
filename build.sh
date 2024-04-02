python3 -m pip install --upgrade build
rm dist/*
python3 -m build
python3 -m pip install dist/*.whl

