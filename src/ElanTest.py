
from pathlib import Path
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


dir_path = Path("data")

txt_files = sorted(dir_path.rglob("*.txt"))  
txt_path = txt_files[0]   #Just grab the first one 

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
#Replace each period with a underscore in the same column

#Create new cololmn ID Gloss ASLEX OLD that is a copy of ID Gloss ASLLEX
df_elan["ID Gloss ASLLEX OLD"] = df_elan["ID Gloss ASLLEX"]

#Truncate ID Gloss ASLLEX at the first period
df_elan["ID Gloss ASLLEX"] = df_elan["ID Gloss ASLLEX"].str.split(".").str[0]





# print(f"Loaded: {txt_path}")
# print(df_elan)
# print("ASL-LEX Data:--------------------------")
# print(ALex_df)
good = df_elan[
    (~df_elan["Response Status"].astype(str).str.contains("Reject", case=False, na=False))
    & (~df_elan["Response Status"].isna())]
bad  = df_elan[df_elan["Response Status"].astype(str).str.contains("Reject", case=False, na=False)]

print("Good trials:")
print(good)

print("\nTrials that had a Rejected response Status -> Discard")
print(bad)




##Dropping the first few rows with NA's and last rows that are empty
entryIDs = set(ALex_df['EntryID'].dropna()) 

def func(row):
    val = row['ID Gloss ASLLEX']
    base = str(val).strip().lower()

    # direct substitutions you will fill in
    subs = {
        "vaccum": "vacuum",
        "climb": "", #Climb_ladder
        "salute": "",
        "shave-head": "", #shave_2
        "mixer": "", #mix 1
        "lasso": "", # No sub
        "zip": "", #No sub
        "hello-gesture": "wave",
        "catch_ca": "",
        "padlock": "lock",
        "matchstick": "",
    }

    # if missing → auto fail
    if pd.isna(val):
        return "", False

    # 1. direct dictionary substitution
    if base in subs:
        fixed = subs[base]
        if fixed in entryIDs:
            return fixed, True
        else:
            return fixed, False   # exists in dict but not in ASL-LEX

    # 2. replace periods with underscores
    gloss = base.replace('.', '_')
    if gloss in entryIDs: 
        return gloss, True
    
    # remove -ca
    gloss = base.split("-ca")[0]
    if gloss in entryIDs:
        return gloss, True

    # remove _ca
    gloss = base.split("_ca")[0]
    if gloss in entryIDs:
        return gloss, True

    # adds _1
    gloss = base + "_1"
    if gloss in entryIDs:
        return gloss, True

    # fail 
    return base, False




good[['CleanGloss', 'Matched']] = good.apply(func, axis=1, result_type='expand')
good.to_csv(dir_path / "PreMerged-PostMod--Elan.csv", index = False)


#Merge
df_joined = good.merge(ALex_df, how='left', left_on="CleanGloss",right_on="EntryID")
df_joined = df_joined[df_joined["Matched"] == True]




# Count how many rows joined and how many didn't
total_rows = len(df_joined)
num_joins_failed = df_joined["EntryID"].isna().sum()

print(f"Total trials after join: {total_rows}")
print(f"Number of joins that failed: {num_joins_failed}")

# print(df_joined)
#Save data frame 
output_path = dir_path / "Elan_ASLLEX_Joined.csv"
df_joined.to_csv(output_path, index=False)
print(f"Saved joined data to: {output_path}")


# To a non parametric T test comparing the "Duration - msec" of lexical classes "Noun" and "Verb"
from scipy.stats import mannwhitneyu
nouns = df_joined[df_joined["LexicalClass"] == "Noun"] #Make a dataframe with just the nouns in it from df_joined
verbs = df_joined[df_joined["LexicalClass"] == "Verb"]
print(f"Nouns: {len(nouns)}, Verbs: {len(verbs)}")
stat, p = mannwhitneyu(nouns["Duration - msec"], verbs["Duration - msec"])
print(f"Mann-Whitney U test between Nouns and Verbs: stat={stat}, p={p}")

results_path = dir_path / "Elan_ASLLEX_Joined_Results.csv"  #Save the results to a csv file that overwrites previous results
with open(results_path, "w") as f:
    f.write("Test,Stat,P-value\n")
print(f"Saved results to: {results_path}")







def plot_duration_nouns_verbs(df_joined):
    # work on a copy
    df = df_joined.copy()

    # keep only matched rows if that column exists
    if "Matched" in df.columns:
        df = df[df["Matched"] == True]

    # keep only Noun / Verb (capitalized, like your data)
    df = df[df["LexicalClass"].isin(["Noun", "Verb"])].copy()
    print(f"Rows after keeping only Noun/Verb: {len(df)}")
    print(df["LexicalClass"].value_counts())

    # clean duration + frequency
    df["duration_msec"] = pd.to_numeric(df["Duration - msec"], errors="coerce")
    df["freq_z"] = pd.to_numeric(df["SignFrequency(Z)"], errors="coerce")
    df = df.dropna(subset=["duration_msec"])
    print(f"Rows after dropping NA durations: {len(df)}")

    # high vs low frequency based on z-score
    df["freq_group"] = df["freq_z"].apply(lambda x: "high" if pd.notna(x) and x >= 0 else "low")

    print("Frequency groups:")
    print(df["freq_group"].value_counts())
    class_order = ["Noun", "Verb"]
    freq_order = ["low", "high"]

    # one big figure with 4 panels
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

     # 1) box: nouns vs verbs
    sns.boxplot(data=df,x="LexicalClass", y="duration_msec",order=class_order,ax=axes[0, 0])
    axes[0, 0].set_title("Duration by Class (box)")
    axes[0, 0].set_xlabel("Lexical Class")
    axes[0, 0].set_ylabel("Duration (msec)")

    # 2) violin: nouns vs verbs
    sns.violinplot( data=df, x="LexicalClass", y="duration_msec", order=class_order, cut=0,ax=axes[0, 1])
    axes[0, 1].set_title("Duration by Class (violin)")
    axes[0, 1].set_xlabel("Lexical Class")
    axes[0, 1].set_ylabel("Duration (msec)")

    # 3) box: class x high/low frequency
    sns.boxplot( data=df, x="LexicalClass", y="duration_msec", hue="freq_group", order=class_order, hue_order=freq_order, ax=axes[1, 0])
    axes[1, 0].set_title("Duration by Class + Freq (box)")
    axes[1, 0].set_xlabel("Lexical Class")
    axes[1, 0].set_ylabel("Duration (msec)")
    axes[1, 0].legend(title="Freq", loc="best")

    # 4) violin: class x high/low frequency
    sns.violinplot( data=df, x="LexicalClass", y="duration_msec", hue="freq_group", order=class_order, hue_order=freq_order, cut=0, split=True, ax=axes[1, 1])
    axes[1, 1].set_title("Duration by Class + Freq (violin)")
    axes[1, 1].set_xlabel("Lexical Class")
    axes[1, 1].set_ylabel("Duration (msec)")
    axes[1, 1].legend(title="Freq", loc="best")

    plt.tight_layout()
    plt.show()

print(df_joined.columns.tolist())
plot_duration_nouns_verbs(df_joined)
