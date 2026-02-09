PROJECT=report
TEX=pdflatex
BIBTEX=bibtex
BUILDTEX=$(TEX) -interaction=nonstopmode $(PROJECT).tex

all: $(PROJECT).pdf

$(PROJECT).pdf: $(PROJECT).tex bibliography.bib
	$(BUILDTEX)
	@grep -q "citation" $(PROJECT).aux && $(BIBTEX) $(PROJECT) || echo "No citations found, skipping BibTeX step."
	$(BUILDTEX)
	$(BUILDTEX)

clean:
	rm -f *.aux *.log *.bbl *.blg *.out *.toc *.pdf

.PHONY: all clean
