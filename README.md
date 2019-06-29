# Getting Started

1. Create a virtual environment and activate it.

```sh
mkdir -p venv
virtualenv --python=`which python3` venv
source venv/bin/activate
```

2. Download the dependencies.
```sh
make get-deps
```

3. Install Pandoc.
```sh
brew install pandoc
```

4. Build and serve the website.
```sh
make serve
```

# Adding new notebooks.

1. Create a new folder under `notebooks/`.

2. Create a `requirements.txt` file for this project and its own virtual environment.

3. Once you're done, just go back to the root directory of this project and run `make serve` to see the conversion to a markdown file.
