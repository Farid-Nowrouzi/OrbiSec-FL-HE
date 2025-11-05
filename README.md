# OrbiSec-FL-HE

Secure federated learning for LEO satellites (Week-1: baseline FedAvg on Digits).  
We simulate 8 clients and a satellite-style dropout scenario, and log accuracy + bytes.

## Run
```bash
pip install -U pip
pip install -r requirements.txt
python experiments/run_week1_baseline.py
