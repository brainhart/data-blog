.PHONY: get-deps serve build source blog

SOURCE := $(wildcard notebooks/*/.)
SOURCE_DIRS := $(patsubst notebooks/%/., site/blog/source/%.tgz, $(SOURCE))
NOTEBOOK_DIRS := $(patsubst notebooks/%/., blog/%, $(SOURCE))

check-env:
ifndef VIRTUAL_ENV
    $(error target must be run inside a virtualenv!)
endif

get-deps: check-env
	pip install -r requirements.txt

serve: build
	make -C site serve

deploy: source blog
	make -C site deploy

build: source blog
	make -C site build

source: $(SOURCE_DIRS)

site/blog/source/%.tgz: notebooks/%/
	mkdir -p site/blog/source/
	tar -czf "$@" -C "$<" .

blog: $(NOTEBOOK_DIRS)

blog/%: notebooks/%
	find "$<" -name \*.ipynb -exec jupyter nbconvert {} --to markdown --output-dir site/blog/ \;
	find "$<" -name \*.png -exec cp {} site/blog \;
	find "$<" -name \*.jpg -exec cp {} site/blog \;
	find "$<" -name \*.gif -exec cp {} site/blog \;

clean:
	make -C site clean
