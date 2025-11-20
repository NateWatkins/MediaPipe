
from pathlib import Path
import pandas as pd


dir_path = Path("../MediaPipe10.20.25Backup")

txt_files = sorted(dir_path.rglob("*.txt"))  
txt_path = txt_files[0]  
csv_files = sorted(dir_path.rglob("signdata (8)(in).csv"))  
csv_path = csv_files[0]  

df_elan= pd.read_csv(txt_path, sep="\t")
ALex_df = pd.read_csv(csv_path)
# Keep the following columns from ASL-LEX data: EntryID, SignFrequency(Z), Code, Iconicity(Z), LexicalClass, Compound.2.0, Handshape.2.0, SelectedFingers.2.0, ThumbPosition.2.0, SignType.2.0, Movement.2.0, MajorLocation.2.0, MinorLocation.2.0, Contact.2.0, Parameter.Neighborhood.Density.2.0
ALex_df = ALex_df[["EntryID", "SignFrequency(Z)", "Code", "Iconicity(Z)", "LexicalClass", "Compound.2.0", "Handshape.2.0", "SelectedFingers.2.0", "ThumbPosition.2.0", "SignType.2.0", "Movement.2.0", "MajorLocation.2.0", "MinorLocation.2.0", "Contact.2.0", "Parameter.Neighborhood.Density.2.0"]]

#Trim the whitespace for both dataframs
#Turn everything in EntryId and ID Gloss ASLLEX to lowercase
ALex_df["EntryID"] = ALex_df["EntryID"].str.strip().str.lower()
df_elan["ID Gloss ASLLEX"] = df_elan["ID Gloss ASLLEX"].str.strip().str.lower()



print(f"Loaded: {txt_path}")
print(df_elan)
print("ASL-LEX Data:--------------------------")
print(ALex_df)
good = df_elan[~df_elan["Response Status"].astype(str).str.contains("Reject", case=False, na=False)]
bad  = df_elan[df_elan["Response Status"].astype(str).str.contains("Reject", case=False, na=False)]

print("Good trials:")
print(good)

print("\nRejected trials:")
print(bad)

df_joined = good.merge(ALex_df, how='left', left_on="ID Gloss ASLLEX",right_on="EntryID")

print(df_joined)
#Save data frame 
output_path = dir_path / "Elan_ASLLEX_Joined.csv"
df_joined.to_csv(output_path, index=False)
print(f"Saved joined data to: {output_path}")
# To a non parametric T test comparing the "Duration - msec" of lexical classes "Noun" and "Verb"
from scipy.stats import mannwhitneyu
#Make a dataframe with just the nouns in it from df_joined
nouns = df_joined[df_joined["LexicalClass"] == "Noun"]
verbs = df_joined[df_joined["LexicalClass"] == "Verb"]
print(f"Nouns: {len(nouns)}, Verbs: {len(verbs)}")
stat, p = mannwhitneyu(nouns["Duration - msec"], verbs["Duration - msec"])
print(f"Mann-Whitney U test between Nouns and Verbs: stat={stat}, p={p}")
