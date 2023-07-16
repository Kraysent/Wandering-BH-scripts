.PHONY: style

style:
	black --line-length 120 .
	isort .

resolution-compute:
	python3 main.py models-resolution -s -st

resolution-save:
	python3 main.py models-resolution -c -s

resolution-show: 
	python3 main.py models-resolution -c

example-save:
	python3 main.py models-example --plot -s

example-show:
	python3 main.py models-example --plot
