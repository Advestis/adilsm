dev_install:
	@pip3 install -e .

clean:
	@find . -name '__pycache__' |xargs rm -fr {} \;
	@rm -fr build dist .eggs .pytest_cache
	@rm -fr adilsm-*.dist-info
	@rm -fr adilsm.egg-info

install: clean wheel
	@pip3 install -U dist/*.whl --cache-dir /pip_cache

wheel: clean
	@python3 -m build --wheel

wheel_upload: clean wheel
	@python3 -m twine upload dist/*
