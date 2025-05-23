# My first RL model

Yipee!

# Developer

0. Setup Ubuntu dependencies:

```bash
sudo apt update
sudo apt install -y python3.10-dev libaio-dev
```

1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create the venv

```bash
uv venv --python 3.10
source .venv/bin/activate
uv sync
```

3. Hugging Face login. Enter token on prompt.

```bash
huggingface-cli login
```

4. Open the Training.ipynb notebook.
