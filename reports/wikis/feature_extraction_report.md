# Feature Extraction Report

This document summarizes the feature extraction process for our recommender system. We merged user, movie, and rating datasets into a single training table and engineered features to represent both user attributes and movie characteristics. Encoding choices were made to balance interpretability, model compatibility, and scalability.

---

## User Features

- **Age (numeric + binned)**  
  The `age` field is preserved as a raw numeric feature and also bucketized into bins (`0–18`, `19–25`, `26–35`, `36–50`, `50+`).  
  *Rationale:* Numeric age allows fine-grained learning, while bins capture broader demographic groups useful for generalization.

- **Occupation (categorical → one-hot)**  
  The `occupation` field is transformed into one-hot encoded vectors.  
  *Rationale:* This prevents introducing an artificial order (e.g., "artist" > "programmer") that could mislead the model. One-hot encoding ensures each occupation is treated independently, allowing the model to learn occupation-specific viewing patterns.

- **Gender (categorical → one-hot)**  
  Gender values include `M` (male), `F` (female), and `U` (unknown). These are one-hot encoded into binary indicators.  
  *Rationale:* One-hot encoding avoids imposing numeric order on gender categories and naturally handles the `unknown` case without bias.


---

## Movie Features

- **Runtime (numeric)**  
  Preserved as continuous values.  
  *Rationale:* Movie length can influence viewing and rating behavior.

- **Popularity, Vote Average, Vote Count (numeric)**  
  Direct numeric values from metadata.  
  *Rationale:* Capture global trends and crowd perception of a movie.

- **Release Year (numeric)**  
  Extracted from `release_date`. Missing years are set to `-1`.  
  *Rationale:* Allows the model to account for temporal effects, e.g., older vs. modern films.

- **Genres (multi-hot encoding)**  
  Split into binary indicator columns (e.g., `genre_Drama=1`, `genre_Comedy=0`).  
  *Rationale:* Genre is a strong predictor of user preference; multi-hot allows multiple genres per movie.

- **Spoken Languages (multi-hot encoding, normalized)**  
  We extracted all spoken languages per movie, normalized them to ISO codes (e.g., `"English"` → `en`, `"日本語"` → `ja`), and created multi-hot binary features (`lang_en`, `lang_ja`, etc.).  
  *Rationale:* Spoken languages reflect linguistic accessibility and cultural reach of a movie.

- **Original Language (categorical → one-hot/label encoding)**  
  Preserved as a single categorical feature (e.g., `"en"`, `"fr"`, `"ja"`).  
  *Rationale:* Indicates the production’s native language, distinct from the multilingual spoken languages in the movie. Encoded separately for more granular signals.

- **Production Countries (multi-hot encoding)**  
  Similar to genres/languages, encoded into binary columns.  
  *Rationale:* Geographical/cultural context often influences style, themes, and audience reception.

---

## Interaction Features

- **Rating (explicit feedback, numeric)**  
  The user-provided rating (1–5 stars) is the primary target label for supervised training.  
  *Rationale:* Ratings are the clearest signal of user preference and directly drive the recommendation objective. 

- **Watch Time (implicit feedback, optional)**  
  Watch time in minutes per movie is available and can be binarized (e.g., watched > X minutes = positive).  
  *Rationale:* While not used as the primary label, watch time provides implicit signals of user engagement and can enrich the model.

---

## Handling Missing Data

- **Numeric Features (runtime, popularity, vote counts, release year):** Missing values imputed with `-1`. This distinguishes missing entries from valid values without distorting distributions.  
- **Categorical Features (original_language, occupation, gender):** Missing values replaced with `"unknown"` category.  
- **Multi-hot Features (genres, spoken_languages, production_countries):** Missing entries default to empty sets, meaning all associated binary indicators remain `0`.

*Rationale:* These strategies preserve all rows for training while ensuring the model can differentiate between “unknown” and true zero/absence cases.

---

## Why Include Both Original and Spoken Languages?

We decided to include both **original language** and **spoken languages** instead of choosing one.  

- **Original language** represents the production context and cultural background of a film. It can be an indicator of thematic style or narrative traditions.  
- **Spoken languages** represent the accessibility and linguistic diversity within the movie itself, e.g., multilingual productions or dubbed versions.  

By including both, the model can learn **cultural origin signals** (from original language) while also capturing **audience accessibility** (from spoken languages). This dual encoding provides richer information for recommendations and reduces the risk of losing nuance in multilingual films.  
