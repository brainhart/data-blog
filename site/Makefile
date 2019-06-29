.PHONY: build clean mkdocs serve

build: mkdocs

mkdocs:
	mkdocs build --site-dir site

clean:
	rm -rf site

serve: build
	mkdocs serve

deploy: build
	mkdocs gh-deploy
