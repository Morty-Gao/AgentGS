# Run `make` here, or `make -C aamas2027` from the project root.
# Ubuntu/WSL dependencies: sudo apt-get install latexmk texlive-latex-base \
#   texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended \
#   texlive-fonts-extra texlive-plain-generic texlive-science
.DEFAULT_GOAL := pdf

LATEXMK ?= latexmk
MAIN := main.tex
OUTDIR := build
LATEXFLAGS := -pdf -interaction=nonstopmode -halt-on-error -file-line-error -synctex=1 -outdir=$(OUTDIR)

.PHONY: all pdf full supplement figures compile watch clean distclean rebuild check-tools help

all: pdf full supplement

compile: pdf

full: check-tools
	$(LATEXMK) $(LATEXFLAGS) full.tex
	@printf 'Full translation: $(OUTDIR)/full.pdf\n'

supplement: check-tools
	$(LATEXMK) $(LATEXFLAGS) supplement.tex
	@printf 'Supplement: $(OUTDIR)/supplement.pdf\n'

FIGURE_SCRIPTS := $(wildcard scripts/fig_*.py)

# Redraw every figure into figures/ (SVG + PDF); needs rsvg-convert or inkscape.
figures:
	@for script in $(FIGURE_SCRIPTS); do python3 $$script || exit 1; done

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

check-tools:
	@command -v $(LATEXMK) >/dev/null || { printf 'Missing latexmk. Install: sudo apt-get install latexmk\n'; exit 1; }
	@command -v pdflatex >/dev/null || { printf 'Missing pdflatex. Install: sudo apt-get install texlive-latex-base\n'; exit 1; }
	@command -v bibtex >/dev/null || { printf 'Missing bibtex. Install: sudo apt-get install texlive-binaries\n'; exit 1; }

help:
	@printf '%s\n' \
	  'make / make pdf / make compile  Compile build/main.pdf and copy to main.pdf (incremental)' \
	  'make full                      Compile the complete Word translation' \
	  'make supplement                Compile the anonymous supplementary figures' \
	  'make all                       Compile all three documents' \
	  'make watch                     Recompile automatically on changes' \
	  'make clean                     Remove auxiliary files; keep PDF' \
	  'make distclean                 Remove generated files, including PDF' \
	  'make rebuild                   Clean and compile from scratch' \
	  'make figures                   Redraw figures from scripts/fig_*.py' \
	  'make help                      Show available commands'
