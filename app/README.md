# Placement outlook app

The web app loads `final_model.pkl`, formats student profiles like the training notebook, and returns a probability, per-student SHAP contributions, and one-input scenario comparisons. The input data is synthetic; scenario results describe model behavior and are not causal claims.

## Run locally on Windows

Run these commands from the repository root in PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r app/requirements.txt
python app/server.py
```

Then open <http://127.0.0.1:5000>. Keep the PowerShell window open while using the app. The app dependency file pins scikit-learn to the 1.6.1 version that saved the model, which avoids the version mismatch warning from the general project requirements.

## Deploy from GitHub with Render

Create a Render **Web Service** from the GitHub repository, using the `main` branch. Leave **Root Directory** blank. Set:

- **Language:** Python 3
- **Build Command:** `pip install -r app/requirements.txt gunicorn`
- **Start Command:** `gunicorn app.server:app --bind 0.0.0.0:$PORT`
- **Instance type:** Free

The root `.python-version` file selects Python 3.13.4. Render deploys the selected Git branch and can automatically deploy new commits pushed to it. Free web services may sleep after inactivity and can take a short time to wake on the next visit.
