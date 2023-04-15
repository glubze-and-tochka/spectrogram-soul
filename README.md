# neuroBot

Bot that will be used as API for NeuroNet that can estimate emotional background of voice message

## Getting Started

- Create virtual environment for Python3.10:

```sh
python3.10 -m venv .venv
```

- For VS Code users: Select Interpreter

- Install dependencies with `poetry`:

```sh
poetry install
```

- After installing dependencies we recommend updating their versions:

```sh
poetry update
```

- To add package or make package production one (packages for data science are
  in `dev` section) use:

```sh
poetry add <package>
```

- More about Poetry commands you can read on their [documentation
  page](https://python-poetry.org/docs/cli/).

- Do not forget to initialize `pre-commit` after installing dependencies:

```sh
pre-commit install --hook-type pre-commit --hook-type pre-push
```

- Add file in source dir with name .env :

```sh
.env
```

Add AIOGRAM_API_TOKEN variable to .env file:

```sh
AIOGRAM_API_TOKEN = {YOUR_SUPER_SECRET_TOKEN}
```

- To start bot https://t.me/xkm555_bot:

```sh
python emo_service/api/bot_run.py
```
