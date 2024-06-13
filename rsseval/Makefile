# ======= FOLDERS ==================
VENV := venv
PROJECT_NAME := mini-boia

# ======= PROGRAMS AND FLAGS =======
PYTHON := python3
PYFLAGS := -m
PIP := pip
UPGRADE_PIP := --upgrade pip

# ======= DOC ======================
AUTHORS := --author "X" 
VERSION :=-r 0.1 
LANGUAGE := --language en
SPHINX_EXTENSIONS := --extensions sphinx.ext.autodoc --extensions sphinx.ext.napoleon --extensions sphinx.ext.viewcode --extensions myst_parser
DOC_FOLDER := docs
COPY := cp -rf
IMG_FOLDER := .github

# ======= FORMAT ===================
FORMAT := black
FORMAT_FLAG := .

## Quickstart
SPHINX_QUICKSTART := sphinx-quickstart
SPHINX_QUICKSTART_FLAGS := --sep --no-batchfile --project bears $(AUTHORS) $(VERSION) $(LANGUAGE) $(SPHINX_EXTENSIONS)

## Build
BUILDER := html
SPHINX_BUILD := make $(BUILDER)
SPHINX_API_DOC := sphinx-apidoc
SPHINX_API_DOC_FLAGS := -P -o $(DOC_FOLDER)/source .
SPHINX_API_DOC_FLAGS_XOR_MNIST := -P -o $(DOC_FOLDER)/source/MNIST XOR_MNIST
SPHINX_API_DOC_FLAGS_BOIA := -P -o $(DOC_FOLDER)/source/BOIA BDD_OIA
SPHINX_THEME = sphinx_rtd_theme
DOC_INDEX := index.html

## INDEX.rst preamble
define INDEX

.. bears documentation master file, created by
   sphinx-quickstart on Tue Feb 20 13:38:46 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. include:: ../../README.md
	 :parser: myst_parser.sphinx_

.. toctree::
   :maxdepth: 8
   :caption: Contents:

   MNIST/modules
   BOIA/modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

endef

export INDEX

# ======= COLORS ===================
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
NONE := \033[0m

# ======= COMMANDS =================
ECHO := echo -e
MKDIR := mkdir -p
OPEN := xdg-open
SED := sed
TOUCH := touch
	
# RULES
.PHONY: help env install install-dev doc-layout open-doc format-code

help:
	@$(ECHO) '$(YELLOW)Makefile help$(NONE)'
	@$(ECHO) " \
	* env 			: generates the virtual environment using the current python version and venv\n \
	* install		: install the requirements listed in requirements.txt\n \
	* install-dev		: install the development requirements listed in requirements.dev.txt\n \
	* doc-layout 		: generates the Sphinx documentation layout\n \
	* doc-layout-no-theme	: generates the Sphinx documentation layout without setting the theme\n \
	* doc 			: generates the documentation (requires an existing documentation layout)\n \
	* format-code 		: formats the code employing black python formatter\n \
	* help 		: prints this message\n \
	* open-doc 		: opens the documentation\n"

env:
	@$(ECHO) '$(GREEN)Creating the virtual environment..$(NONE)'
	@$(MKDIR) $(VENV)
	@$(eval PYTHON_VERSION=$(shell $(PYTHON) --version | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]' | cut -f1,2 -d'.'))
	@$(PYTHON_VERSION) -m venv $(VENV)/$(PROJECT_NAME)
	@$(ECHO) '$(GREEN)Done$(NONE)'

install:
	@$(ECHO) '$(GREEN)Installing requirements..$(NONE)'
	@$(PYTHON) -m pip install $(UPGRADE_PIP)
	@pip install -r requirements.txt
	@$(ECHO) '$(GREEN)Done$(NONE)'

install-dev:
	@$(ECHO) '$(GREEN)Installing requirements..$(NONE)'
	@$(PYTHON) -m pip install $(UPGRADE_PIP)
	@$(PIP) install -r requirements.dev.txt
	@$(ECHO) '$(GREEN)Done$(NONE)'

doc-layout:
	@$(ECHO) '$(BLUE)Generating the Sphinx layout..$(NONE)'
	# Sphinx quickstart
	@$(SPHINX_QUICKSTART) $(DOC_FOLDER) $(SPHINX_QUICKSTART_FLAGS)
	# Including the path for the current README.md
	@echo "import os" >> $(DOC_FOLDER)/source/conf.py
	@echo "import sys" >> $(DOC_FOLDER)/source/conf.py
	@echo "sys.path.insert(0, os.path.abspath('../..'))" >> $(DOC_FOLDER)/source/conf.py
	# Inserting custom index.rst header
	@echo "$$INDEX" > $(DOC_FOLDER)/source/index.rst
	# Sphinx theme
	@$(SED) -i -e "s/html_theme = 'alabaster'/html_theme = '$(SPHINX_THEME)'/g" $(DOC_FOLDER)/source/conf.py 
	# Add .nojekyll
	@$(TOUCH) .nojekyll
	@echo "html_extra_path = ['../../.nojekyll', '../../README.md']" >> $(DOC_FOLDER)/source/conf.py
	# Copy the image folder inside the doc folder
	@$(COPY) $(IMG_FOLDER) $(DOC_FOLDER)/source
	@$(ECHO) '$(BLUE)Done$(NONE)'

doc:
	@$(ECHO) '$(BLUE)Generating the documentation..$(NONE)'
	$(SPHINX_API_DOC) $(SPHINX_API_DOC_FLAGS)
	$(SPHINX_API_DOC) $(SPHINX_API_DOC_FLAGS_XOR_MNIST)
	$(SPHINX_API_DOC) $(SPHINX_API_DOC_FLAGS_BOIA)
	cd $(DOC_FOLDER); $(SPHINX_BUILD)
	@$(ECHO) '$(BLUE)Done$(NONE)'

open-doc:
	@$(ECHO) '$(BLUE)Open documentation..$(NONE)'
	$(OPEN) $(DOC_FOLDER)/build/$(BUILDER)/$(DOC_INDEX)
	@$(ECHO) '$(BLUE)Done$(NONE)'

format-code:
	@$(ECHO) '$(BLUE)Formatting the code..$(NONE)'
	@$(FORMAT) $(FORMAT_FLAG)
	@$(ECHO) '$(BLUE)Done$(NONE)'
