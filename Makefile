.PHONY: style

style:
	black --line-length 120 .
	isort .
