import pandas as pd
import unidecode
from statsmodels.stats.outliers_influence import variance_inflation_factor
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter    # For debugging purposes

# Count number of concussions sustained in a season. Also count total games missed per season.
def cleanConcussionData(df):
    df = df.groupby(
        ["Season", "Player"], as_index=False).agg(
            {
                "Games Missed": "sum",
                "Injury Type": "count"
            }
        ).rename(
            columns={"Injury Type": "Number of Concussions"}
        )
    df["Player"] = df["Player"].apply(standardizeName)
    return df


def cleanPerformanceData(path):
    df = pd.concat(
        map(pd.read_csv, [f"{path}20{i}-20{i+1}.csv" for i in range(12, 23)]), ignore_index = True)
    if path == "data/playerStats/":
        fiveOnFive = df[df["situation"] == "5on5"]
        df = df[df["situation"] == "all"]
        df = pd.merge(
            df, 
            fiveOnFive[["name", "season", "xGoalsForAfterShifts", "xGoalsAgainstAfterShifts", "corsiForAfterShifts", "corsiAgainstAfterShifts", "fenwickForAfterShifts", "fenwickAgainstAfterShifts"]], 
            on=["name", "season"], 
            how="left", 
            suffixes=("", "_5v5")
        )
        # Fixing incorrectly reported stats in the dataset
        df["xGoalsForAfterShifts"] = df["xGoalsForAfterShifts_5v5"]
        df["xGoalsAgainstAfterShifts"] = df["xGoalsAgainstAfterShifts_5v5"]
        df["corsiForAfterShifts"] = df["corsiForAfterShifts_5v5"]
        df["corsiAgainstAfterShifts"] = df["corsiAgainstAfterShifts_5v5"]
        df["fenwickForAfterShifts"] = df["fenwickForAfterShifts_5v5"]
        df["fenwickAgainstAfterShifts"] = df["fenwickAgainstAfterShifts_5v5"]
        df = df.drop(columns=["xGoalsForAfterShifts_5v5", "xGoalsAgainstAfterShifts_5v5", "corsiForAfterShifts_5v5", "corsiAgainstAfterShifts_5v5", "fenwickForAfterShifts_5v5", "fenwickAgainstAfterShifts_5v5"])
    else: 
        df = df[df["situation"] == "all"]
    df["Season"] = df["season"].apply(lambda x: f"{x}/{str(x + 1)[-2:]}")
    df["Player"] = df["name"].apply(standardizeName)
    df["position"] = df["position"].apply(mapPosition)
    return df


# Make the names in both datasets the same format
def standardizeName(name):
    if "," in name:
        last, first = name.split(", ")
        name = f"{first} {last}"
    return unidecode.unidecode(name)


# Change position feature to a number
def mapPosition(position):
    match position:
        case 'C':
            return 1
        case 'L':
            return 2
        case 'R':
            return 3
        case 'D':
            return 4
        case _:
            return 0


# Merge concussion and performance dataframes into one dataframe
def mergeDatasets(df1, df2):
    df = pd.merge(
        df1,
        df2,
        how='left',
        left_on=["Season", "Player"],
        right_on=["Season", "Player"]
    )
    df["Games Missed"] = df["Games Missed"].fillna(0).astype(int)
    df["Number of Concussions"] = df["Number of Concussions"].fillna(0).astype(int)
    df.drop(columns=["Season", "Player", "situation"], inplace=True)
    df["season"] = df["season"].apply(lambda x: f"{x}/{str(x + 1)[-2:]}")
    return df


# Gets the VIF for every feature in the dataframe
def getVIF(df):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i)
                          for i in range(len(df.columns))]
    vif_data = vif_data[vif_data["VIF"].notna()]
    return vif_data


# Iteratively remove features until max VIF is < 10
def removeMulticollinearity(df):
    while True:
        vif_data = getVIF(df)
        maxVIF = vif_data["VIF"].max()
        if maxVIF < 10:
            break
        dropFeature = vif_data.loc[vif_data["VIF"].idxmax(), "feature"]
        df = df.drop(columns=[dropFeature])
        print(f"Dropped {dropFeature} with VIF {maxVIF}")
    return df, vif_data


def getTrainingSetandLabels():
    # Dataset containing only NHL players that sustained concussions 2012-2023
    nhl_concussions = pd.read_csv("data/concussionPlayers/dataset/NHL Concussions Database_data.csv")
    nhl_concussions = nhl_concussions.loc[0:, 'Season':'Games Missed']
    nhl_concussions = nhl_concussions[~nhl_concussions["Season"].str.contains(r"\(playoffs", na=False)]
    nhl_concussions_skaters = nhl_concussions[nhl_concussions["Position"] != "G"]
    nhl_concussions_goalies = nhl_concussions[nhl_concussions["Position"] == "G"]
    nhl_concussions_skaters = cleanConcussionData(nhl_concussions_skaters)
    nhl_concussions_goalies = cleanConcussionData(nhl_concussions_goalies)

    # Dataset containing NHL player performance stats 2012-2023
    nhl_stats_skaters = cleanPerformanceData("data/playerStats/")
    nhl_stats_goalies = cleanPerformanceData("data/playerStats/goalie-")

    # Compile nhl_concussions and nhl_stats into one dataset each for skaters and goalies.
    nhl_skaters = mergeDatasets(nhl_stats_skaters, nhl_concussions_skaters)
    nhl_goalies = mergeDatasets(nhl_stats_goalies, nhl_concussions_goalies)

    # Add target feature column for next season concussion. Drop the 2022/23 season because we have no target features for it.
    nhl_skaters["Concussion next-season"] = (
        nhl_skaters.groupby("name")["Number of Concussions"].shift(-1).fillna(0).astype(int)
    )
    nhl_skaters["Concussion next-season"] = (nhl_skaters["Concussion next-season"] > 0).astype(int)
    last_season_year = nhl_skaters["season"].str.split("/").str[0].astype(int).max()
    nhl_skaters = nhl_skaters[nhl_skaters["season"].str.split("/").str[0].astype(int) < last_season_year]

    # Feature selection is done by detecting multicollinearity via VIF.
    # The removeMulticollinearity function takes a long time to run,
    # so I will manually drop the multicollinear features after running the function once to figure out the features to drop.
    # nhl_skaters, vif_data = removeMulticollinearity(nhl_skaters.loc[0:, "position":])
    X_skaters = nhl_skaters[["position", "iceTimeRank", "I_F_secondaryAssists", "I_F_reboundGoals", "I_F_playStopped", 
                            "penalties", "I_F_hits", "I_F_takeaways", "I_F_lowDangerGoals", "I_F_mediumDangerGoals",
                            "I_F_highDangerGoals", "I_F_dZoneGiveaways", "I_F_xGoalsFromActualReboundsOfShots", 
                            "faceoffsWon", "penalityMinutesDrawn", "shotsBlockedByPlayer", "OnIce_A_reboundGoals",
                            "xGoalsForAfterShifts", "xGoalsAgainstAfterShifts", "Games Missed", "Number of Concussions"]].values    # Training samples
    y_skaters = nhl_skaters["Concussion next-season"].values    # Target feature

    # Fixing imbalanced dataset with SMOTE and random undersampling
    # Minority will have 10% of num of examples of majority first, then majority will have 50% of the num of examples of minority
    overSample = SMOTE(sampling_strategy=0.1, random_state=42)
    underSample = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    steps = [('o', overSample), ('u', underSample)]
    pipeline = Pipeline(steps=steps)
    X_skaters, y_skaters = pipeline.fit_resample(X_skaters, y_skaters)

    return X_skaters, y_skaters


def main():
    X_skaters, y_skaters = getTrainingSetandLabels()


if __name__=='__main__':
    main()


# FOR DEBUGGING

# print(nhl_stats_skaters.head())
# print(nhl_stats_skaters.shape)
# print(nhl_skaters[nhl_skaters["name"] == "Ondrej Kase"])
# print(nhl_skaters.loc[0:, "games_played":].head())

# print(vif_data.info())
# print(vif_data.iloc[0:, 0:])
# print(vif_data.shape)
# print(nhl_skaters.info())
# print(nhl_skaters.head())
# print(nhl_skaters.shape)

# print(nhl_stats_goalies.shape)
# print(nhl_goalies[nhl_goalies["name"] == "Jonas Gustavsson"])
# print(nhl_goalies.head())
# print(nhl_goalies.shape)