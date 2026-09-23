# Getting Started

## Environment Setup

This project requires Python 3.11. You can set up the environment using either `pip` with `virtualenv`/`venv`, or using `conda`.

### Option 1: Using pip and venv (Recommended for non-Conda users)

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
2. Activate the virtual environment:
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Using Conda

1. Create the Conda environment from the provided configuration file:
   ```bash
   conda env create -f environment.yml
   ```
2. Activate the environment:
   ```bash
   conda activate flashbalance
   ```

### Verification
To verify that the environment is set up successfully, run the following command. It should exit without any errors and print `OK`.
```bash
python -c "from stable_baselines3 import PPO; print('OK')"
```
