# Run `make` here, or `make -C aamas2027` from the project root.
# Ubuntu/WSL dependencies: sudo apt-get install latexmk texlive-latex-base \
#   texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended \
#   texlive-fonts-extra texlive-plain-generic texlive-science
.DEFAULT_GOAL := pdf

LATEXMK ?= latexmk
MAIN := main.tex
OUTDIR := build
LATEXFLAGS := -pdf -interaction=nonstopmode -halt-on-error -file-line-error -synctex=1 -outdir=$(OUTDIR)

.PHONY: all pdf full supplement word figures compile watch clean distclean rebuild import-assets check-tools help

all: pdf full supplement

compile: pdf

full: check-tools
	$(LATEXMK) $(LATEXFLAGS) full.tex
	@printf 'Full translation: $(OUTDIR)/full.pdf\n'

supplement: check-tools
	$(LATEXMK) $(LATEXFLAGS) supplement.tex
	@printf 'Supplement: $(OUTDIR)/supplement.pdf\n'

word: full
	python3 scripts/export_word.py

figures:
	python3 scripts/export_drawio.py

pdf: check-tools
	$(LATEXMK) $(LATEXFLAGS) $(MAIN)
	cp "$(OUTDIR)/main.pdf" main.pdf
	@printf 'PDF: main.pdf\n'

# Continuously rebuild when source files change; Ctrl+C stops watching.
watch: check-tools
	$(LATEXMK) $(LATEXFLAGS) -pvc -view=none $(MAIN)

# Remove auxiliary files while keeping the compiled PDF.
clean: check-tools
	$(LATEXMK) -c -outdir=$(OUTDIR) $(MAIN)

# Remove generated files, including the compiled PDF.
distclean: check-tools
	$(LATEXMK) -C -outdir=$(OUTDIR) $(MAIN)
	rm -f main.pdf

# Recursive calls keep cleaning and compiling sequential, even with make -j.
rebuild:
	$(MAKE) distclean
	$(MAKE) pdf

# Import an explicitly selected manuscript; no stale source is silently used.
import-assets:
	@test -n "$(SOURCE)" || { printf 'Usage: make import-assets SOURCE=/path/to/manuscript.docx\n'; exit 1; }
	python3 scripts/import_docx_assets.py "$(SOURCE)"

check-tools:
	@command -v $(LATEXMK) >/dev/null || { printf 'Missing latexmk. Install: sudo apt-get install latexmk\n'; exit 1; }
	@command -v pdflatex >/dev/null || { printf 'Missing pdflatex. Install: sudo apt-get install texlive-latex-base\n'; exit 1; }
	@command -v bibtex >/dev/null || { printf 'Missing bibtex. Install: sudo apt-get install texlive-binaries\n'; exit 1; }

help:
	@printf '%s\n' \
	  'make / make pdf / make compile  Compile build/main.pdf and copy to main.pdf (incremental)' \
	  'make full                      Compile the complete Word translation' \
	  'make supplement                Compile the anonymous supplementary figures' \
	  'make word                      Export the complete translation to Word' \
	  'make all                       Compile all three documents' \
	  'make watch                     Recompile automatically on changes' \
	  'make clean                     Remove auxiliary files; keep PDF' \
	  'make distclean                 Remove generated files, including PDF' \
	  'make rebuild                   Clean and compile from scratch' \
	  'make figures                   Export editable diagrams (requires drawio)' \
	  'make import-assets SOURCE=...  Extract figures from a selected DOCX' \
	  'make help                      Show available commands'
