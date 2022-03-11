# btc_research

This project is optimized to run as a remote container using vscode

### Setup (currently supported only on MACOS, assuming you have homebrew installed)

- Install `VSCode`
```bash
brew install --cask vscode
```

- Install the `Docker Desktop`
```bash
brew install --cask docker
```
- open this folder using VSCode, VSCode will suggest running this folder as a `dev container`, let it

Once you open this folder as a dev container the first time, it will automatically run the following setup:
- Create a Docker image using Dockerfile.dev which includes latest pip and virtualenv
- Install the following VSCode extensions under the dev container: `Python`, `Jupyter`
- Create a virtualenv in the venv folder and switch
- Install all the dependecies given in `requirements.txt`

### Running streamlitApp_1.py
```bash
streamlit run streamlitApp_1.py
```
