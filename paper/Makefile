SHELL="/bin/bash"

TITLE="final_report"
TITLE_test="test"

all:
	pdflatex $(TITLE).tex
	bibtex $(TITLE)
	pdflatex $(TITLE).tex
	pdflatex $(TITLE).tex

test:
	pdflatex $(TITLE_test).tex
	bibtex $(TITLE_test)
	pdflatex $(TITLE_test).tex
	pdflatex $(TITLE_test).tex

clean:
	rm -f *.{aux,log,bbl,lof,lot,blg,out}
