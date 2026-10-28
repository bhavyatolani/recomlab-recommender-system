# RecomLab — Recommender Analytics

A Streamlit learning workspace for exploring recommender systems: upload user–item interaction data (or use a built-in demo dataset), pick an algorithm label and a test split, and browse dataset snapshots, evaluation-style metrics, and personalized / global item rankings — plus a Learn tab that explains the workflow.

> **Read this first — this is a demo prototype, not a trained recommender.**
>
> The app does **not** train Collaborative Filtering, SVD, NeuMF, or LightGCN/NGCF. It runs a simulator called `MockRecSys`: the RMSE / Precision@K / Recall@K values are **preset illustrative numbers**, and the "personalized" rankings are **random items with random scores**, seeded only by the digits in the user id. The only ranking actually computed from your data is the **global** top-items list (average rating in the training split). Do not quote the preset metrics or the personalized rankings as experimental results. See [What this app does — and does not do](#what-this-app-does--and-does-not-do) and [Limitations](#limitations).

## Screenshots

<!-- ADD SCREENSHOT 1 HERE: Overview tab after running the demo -->
![Overview tab — add your screenshot here](<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/4b0b3504-aaee-4286-914c-40bed028fa55" />
)

<!-- ADD SCREENSHOT 2 HERE: Evaluation tab -->
![Evaluation tab — add your screenshot here](screenshots/02-evaluation.png)

<!-- ADD SCREENSHOT 3 HERE: Recommendations tab, personalized view -->
![Recommendations tab — add your screenshot here](screenshots/03-recommendations.png)

<!-- ADD SCREENSHOT 4 HERE: Learn tab -->
![Learn tab — add your screenshot here](screenshots/04-learn.png)

> To add screenshots: run the app, take a screenshot of each tab, save it under `screenshots/` with the name above, and commit it. Or, in the GitHub web editor, delete an image line above and drag your screenshot into its place.

## What this app does — and does not do

| | In this app |
|---|---|
| ✅ Does | Loads and validates a `User_ID, Item_ID, Rating` CSV; generates small built-in demo datasets; splits rows into train/test; shows dataset counts and a data preview; shows a global ranking computed from training-data average ratings; explains metrics and algorithms in the Learn tab; keeps results in `st.session_state` and marks them stale when settings change. |
| ❌ Does not do | Train any recommender model; learn user/item embeddings; predict ratings; compute RMSE, Precision@K, or Recall@K from held-out data; personalize rankings from a user's history; use the test split for evaluation. The test split is only counted and stored. |

## Features

- **Four tabs:** Overview, Evaluation, Recommendations, and Learn.
- **Sidebar workspace:** experience level (Novice / Expert), domain or custom CSV upload, test-split slider, algorithm selector, and Top-K selector.
- **Built-in demo domains:** University Elective Papers, Fashion Recommendation, and Restaurant Selection — each 180 synthetic rows, 5 items.
- **Custom CSV upload:** validated before anything is shown; an invalid file stops the app with an error instead of silently falling back to demo data.
- **Stale-result detection:** if the dataset, model, test split, Top-K, or domain changes after a run, the app warns that the displayed results are out of date until you run again.
- **Novice / Expert modes:** Expert mode adds a technical-details block on the Evaluation tab.
- **Custom professional theme:** dark sidebar, gradient hero, DM Sans / Manrope typography, card-style metrics and recommendation rows (loaded from Google Fonts — needs internet for the fonts; the app still works without them).

## How it works

1. **Data in.** Either a built-in demo dataset is generated (`make_demo_data`, seed 42) or a CSV is uploaded and cleaned by `validate_dataset` (see [Your CSV](#your-csv)).
2. **Split.** On clicking **Run recommendation demo**, rows are split with `sklearn.model_selection.train_test_split`, `random_state=42`, at the slider percentage (10–40%, default 20%). This is a **random row split** — the same user and item can appear on both sides.
3. **"Fit".** `MockRecSys.fit()` only collects the sorted unique user ids and item ids from the training rows. No parameters are learned.
4. **Display.** The Overview and Evaluation tabs show the preset metrics for the selected label. The Recommendations tab shows either the simulator's random personalized list or the global list computed from training data. The result is stored in `st.session_state["recomlab_result"]` together with a dataset signature, so later setting changes can be detected.

### The simulator, in detail

```text
MockRecSys
├── fit(df)                  → remembers sorted unique User_IDs and Item_IDs only
├── get_macro_metrics()      → returns a PRESET dict for the selected label
├── recommend_for_user()     → random permutation of items, scores ~ Uniform(0.70, 0.99)
│                              seeded by the digits in the user id (or 42)
└── get_global_top_items()   → REAL: group training rows by Item_ID,
                               mean Rating, count as tie-breaker, take Top-K
```

### Preset metrics — illustrative only

These values are hard-coded in `MockRecSys.get_macro_metrics()` and are identical on every run, for every dataset. They are **not** measured, and they will always rank LightGCN/NGCF first and Collaborative Filtering last, by construction.

| Algorithm label | RMSE | Precision@K | Recall@K |
|---|---:|---:|---:|
| Collaborative Filtering | 0.84 | 76.5% | 71.2% |
| Matrix Factorization (SVD) | 0.72 | 82.1% | 78.4% |
| Neural Matrix Factorization (NeuMF) | 0.65 | 88.9% | 84.1% |
| LightGCN / NGCF | 0.59 | 92.3% | 89.7% |

The app says this on screen, in the Overview caption, the Evaluation caption, and the sidebar footer. Keep those warnings if you fork the app.

### Personalized vs global — know the difference

| View | How it is produced | Can you trust it? |
|---|---|---|
| **Personalized for a user** | Random permutation of the training items, seeded by the digits in the selected user id; scores drawn from Uniform(0.70, 0.99) and sorted descending. The user's own ratings are never read. | No — illustrative layout only. Two users whose ids share the same digits get the same ordering. |
| **Global popular items** | Mean `Rating` per `Item_ID` over the training rows, ties broken by interaction count, Top-K taken. | Yes, as a descriptive statistic of the training split — it is an average-rating ranking, not a model prediction. |

## Your CSV

Required columns, with these exact names:

| Column | Meaning |
|---|---|
| `User_ID` | User identifier (repeated across rows is expected) |
| `Item_ID` | Item identifier (repeated across rows is expected) |
| `Rating` | Numeric rating or interaction score |

Example:

```csv
User_ID,Item_ID,Rating
User_1,Item_101,5
User_1,Item_104,3
User_2,Item_101,4
```

Validation rules in `validate_dataset()`:

- The file must not be empty, and all three columns must be present.
- Rows with a missing `User_ID`, `Item_ID`, or `Rating`, and rows whose `Rating` is not numeric, are dropped. Only the three required columns are kept.
- After cleaning there must be at least 5 rows, at least 2 unique users, and at least 2 unique items — otherwise the app stops with an error message.

Notes:

- Extra columns in your CSV are ignored (only the three required columns are kept).
- If you choose **Custom Upload** and do not upload a file, the app uses the built-in generic demo dataset (25 items, `Item_500`–`Item_524`) and labels it "Built-in demo dataset".
- There is no file-size handling in the code beyond Streamlit's own upload limit; very large CSVs will be slow because the whole file is read into memory.

## Built-in demo datasets

All demo datasets have 180 rows, are generated with seed 42, and assign each row a random item from the domain's 5 items (so item counts are roughly even, not realistic popularity).

| Domain | Users | Items | Ratings |
|---|---|---|---|
| University Elective Papers | `Student_1`–`Student_29` | Advanced AI, Cloud Computing, Quantum Physics, Blockchain, Cyber Security | 3–5 |
| Fashion Recommendation | `Shopper_1`–`Shopper_29` | Denim Jacket, Running Shoes, Leather Boots, Casual T-Shirt, Summer Dress | 1–5 |
| Restaurant Selection | `Diner_1`–`Diner_29` | Sushi House, Pizza Garden, Burger Joint, Taco Express, Tandoori Night | 1–5 |
| Custom Upload (no file) | `User_100`–`User_129` | `Item_500`–`Item_524` (25 items) | 1–5 |

## Controls

Everything is in the sidebar:

| Control | Options / range | Default |
|---|---|---|
| Experience level | Novice (Simple Results), Expert (Detailed Mechanics) | Novice |
| Choose a domain | Custom Upload, University Elective Papers, Fashion Recommendation, Restaurant Selection | Custom Upload |
| Upload interaction data | CSV only, shown for Custom Upload | — |
| Test split | 10–40%, step 5% | 20% |
| Recommendation algorithm | Collaborative Filtering, Matrix Factorization (SVD), Neural Matrix Factorization (NeuMF), LightGCN / NGCF | Collaborative Filtering |
| Number of recommendations (K) | 3, 5, 10 | 5 |
| Run recommendation demo | Button — nothing is computed until you click it | — |

Changing the experience level only changes how much detail the Evaluation tab shows; it does not change any number.

## The four tabs

### 📊 Overview

Dataset status (source, user count, interaction count), the preset metric cards once a run is current, a dataset snapshot (rows, unique users/items, train/test counts), and a 10-row preview. Before the first run it shows a 3-step getting-started card and the same preview.

### 📈 Evaluation

The current configuration (algorithm, domain, train/test rows, split, K), the preset metrics as a table, and — in Expert mode — a technical-details block naming the split strategy and the `MockRecSys (simulated)` engine, plus a note on what a genuine evaluation would require.

### ✨ Recommendations

Two perspectives, chosen by radio button:

- **Personalized for a user** — pick a user from the training set and see the simulator's Top-K list with "Match score" badges.
- **Global popular items** — the training-data average-rating ranking described above, shown as a ranked table.

### 📘 Learn

Four expanders: (1) what a user–item interaction dataset is, with a CSV example; (2) what RMSE, Precision@K, and Recall@K mean, and that the last two need a relevance threshold and protocol; (3) what each algorithm label represents, ending with "the current application only selects algorithm labels; it does not implement or train these algorithms"; (4) the CSV column requirements.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app (from this folder)
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501). No data files, API keys, or environment variables are needed — demo data is generated in code.

### Deploy on Streamlit Community Cloud

1. Push this folder to a public GitHub repo.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing at the repo, branch `main`, main file `app.py`.
3. Deploy. `requirements.txt` is picked up automatically.

## Files

```
.
├── README.md
├── app.py               # the whole app: theme, MockRecSys, data helpers, UI
├── requirements.txt
├── .gitignore
└── screenshots/         # add your screenshots here (optional, see above)
```

## Requirements

- Python 3.9+
- Streamlit
- pandas
- NumPy
- scikit-learn (only for `train_test_split`)

```bash
pip install -r requirements.txt
```

No version pins are set in `requirements.txt`, because the code does not pin any — install the current releases and, if Streamlit changes an API used here, pin from the error you get. The code was written against the current Streamlit widget APIs (`st.select_slider`, `st.container(border=True)`, `st.tabs`).

## Making it a real recommender

The app is structured so the simulator can be swapped out without redesigning the UI. The honest next steps, in order:

1. **Replace `MockRecSys`** with a real estimator that implements `fit(train_df)` and `predict(user_id, item_id)` — e.g. a bias/SVD model, or a neural model trained on the training split only.
2. **Compute the metrics from `test_df`.** RMSE from predicted vs actual test ratings; Precision@K / Recall@K from ranked test items with an explicit relevance threshold (e.g. rating ≥ 4) and a stated protocol (which items are candidates, how training items are excluded).
3. **Make recommendations from the model**, not from a random permutation: score candidate items the user has not seen in training, rank, take Top-K.
4. **Fix the split for evaluation.** A random row split lets the same user/item appear on both sides and is fine for a first demo, but report it as such. For a cleaner claim, use a per-user hold-out or a time-based split if timestamps exist.
5. **Delete or clearly label the preset metrics dict** once real metrics exist — leaving both in the code invites the preset numbers to be quoted by mistake.
6. **Add tests** for `validate_dataset` and for the new metric functions, including the edge cases the validator already guards (fewer than 2 users/items, fewer than 5 rows, non-numeric ratings).

Until steps 1–3 are done, keep every "simulated / illustrative / demo" warning in the UI and in this README.

## Limitations

- **No model is trained.** All algorithm comparison in the UI is preset values; the app cannot tell you which algorithm is actually better on your data.
- **Personalized rankings are random.** They ignore the selected user's ratings entirely.
- **The test split is unused for evaluation.** It only affects the train/test counts and which rows feed the global ranking and the user/item lists.
- **Random row split.** Users and items leak across train/test; results from a future real model on this split would be optimistic for cold-start claims.
- **Synthetic demo data.** 180 rows, 5 items per domain, near-uniform item assignment, and (for electives) ratings only 3–5 — nothing here resembles a real interaction distribution.
- **No persistence, accounts, or sharing.** Results live in the browser session only; refreshing or changing the dataset signature clears the "current" state.
- **No handling of duplicates, timestamps, or implicit feedback.** Duplicate (user, item) rows are kept as-is; there is no timestamp column; a rating of 1 is treated the same way as any other numeric rating.
- **Fonts need internet.** The DM Sans / Manrope theme is imported from Google Fonts; offline, the app falls back to system fonts.
- **Accessibility / mobile.** The custom CSS targets Streamlit's internal `data-testid` attributes, which can change between Streamlit releases; the layout is usable on narrow screens but the sidebar-heavy design is desktop-first.

## Checked

Verified against the pasted code before writing this README: the file compiles (`python -m py_compile` passes); the preset metric values in the table above match `MockRecSys.get_macro_metrics()` exactly; `recommend_for_user` seeds its RNG from the user-id digits and draws scores from Uniform(0.70, 0.99); `get_global_top_items` groups by `Item_ID` on mean rating with count as tie-breaker; `test_df` is stored and counted but never used to compute a metric; the demo datasets, validation rules, sidebar controls, and tab contents are all read from the code. No run of the app was performed and no screenshot was taken, so the screenshot slots above are placeholders. The code in this repo is the pasted code as supplied; no behaviour was changed.

