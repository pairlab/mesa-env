# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?= -E
SPHINXBUILD   ?= sphinx-build
# SOURCEDIR should contain conf.py and all .md doc files
SOURCEDIR     = .
BUILDDIR      = _build
APIDOC        = sphinx-apidoc
# APIDIR is the directory where the API docs are generated
APIDIR		  = api
# PACKAGE is the package to document API
PACKAGE       = ../mimiclabs

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

apidoc:
	@$(APIDOC) $(PACKAGE) -o "$(APIDIR)" --no-toc --force --module-first

# Prepare the HTML files before deploying
prep:	
	python prepare_html.py

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
