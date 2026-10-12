import pandas as pd
df = pd.read_csv("results/exp4_revised_nstdb_predictions.csv")
clean_df = df[df["condition"] == "clean_interval"]
cols = [c for c in df.columns if "p_usable" in c]
for c in cols:
    print(f"{c}: {clean_df[c].mean():.3f} ± {clean_df[c].std():.3f}")
