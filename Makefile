format:
	pre-commit run --all-files

unit-tests:
	pytest libs/core/tests/unit_tests
