.PHONY: test quick full report clean
test:    ; python -m tests.test_all && python -m tests.test_datasets
quick:   ; python run_all.py --quick
full:    ; python run_all.py --full
report:  ; latexmk -pdf -interaction=nonstopmode main.tex
clean:   ; rm -rf results figures __pycache__ tests/__pycache__
