#!/bin/bash
pdflatex main.tex
bibtex main || true
pdflatex main.tex
pdflatex main.tex
