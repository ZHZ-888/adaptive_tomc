# caculate the average time consumption of each platoon type, and compare the difference between two datasets

r1 = {'ravh40': ['AH', 15.1, 28.8], 'ravh180': ['AHH', 29.1, 44.7], 'ravh340': ['AHHHH', 45.2, 64.6], 'ravh450': ['AHH', 59.2, 74.8], 'ravh580': ['AH', 69.5, 83.2], 'ravh700': ['AHHH', 81.1, 98.6], 'ravh890': ['AHHHA', 100.1, 118.7], 'ravh1140': ['AHHHH', 125.2, 144.5], 'ravh1290': ['A', 140.3, 151.9], 'ravh1440': ['AHHHH', 155.1, 174.3], 'ravh1770': ['AHHHH', 188.1, 207.4], 'ravh1920': ['AH', 203.3, 217.0], 'ravh2090': ['AH', 220.1, 233.8], 'ravh2210': ['AHHHA', 232.1, 250.7], 'ravh2410': ['A', 252.2, 263.9], 'ravh2640': ['AHHHH', 275.2, 294.5], 'ravh2750': ['A', 289.2, 300.8], 'ravh2990': ['AHHHHHHH', 310.1, 334.5], 'ravh3350': ['AHHHH', 346.1, 365.4], 'ravh3540': ['AHHH', 365.1, 382.6], 'ravh3800': ['AHH', 391.1, 406.7], 'ravh4080': ['AHHHH', 419.1, 438.3], 'ravh4290': ['AH', 440.1, 453.8], 'ravh4430': ['AHH', 454.1, 469.7], 'ravh4540': ['AHH', 465.3, 480.9], 'ravh4680': ['AH', 479.1, 492.8], 'ravh4770': ['AHH', 488.2, 503.9], 'ravh4970': ['AHHH', 508.1, 525.6], 'ravh5220': ['AAH', 533.1, 548.2], 'ravh5580': ['AHHHHH', 569.1, 590.1], 'ravh5810': ['AHHH', 592.1, 609.6], 'ravh5930': ['AHH', 604.6, 620.2], 'ravh6070': ['AHH', 618.1, 633.7], 'ravh6210': ['AHHH', 632.1, 649.6], 'ravh6380': ['AHHHHH', 649.1, 670.1], 'ravh6500': ['AHH', 665.1, 680.8], 'ravh6660': ['AHHHHHHH', 677.2, 701.6], 'ravh6810': ['AHH', 697.0, 712.6], 'ravh7070': ['AHH', 718.1, 733.7], 'ravh7230': ['AHHHH', 734.3, 753.6], 'ravh7510': ['AHHH', 762.1, 779.6], 'ravh7670': ['AHHH', 778.1, 795.6], 'ravh7770': ['AH', 790.0, 803.7], 'ravh7980': ['AH', 809.1, 822.9], 'ravh8130': ['AHHHHH', 824.1, 845.1], 'ravh8270': ['AHHH', 840.0, 857.5], 'ravh8520': ['AHH', 863.1, 878.8], 'ravh8640': ['AHH', 875.2, 890.8], 'ravh8860': ['AH', 897.1, 910.8], 'ravh8960': ['AHH', 907.2, 922.8], 'ravh9120': ['AH', 923.1, 936.8], 'ravh9200': ['AH', 931.4, 945.1], 'ravh9410': ['AHH', 952.1, 967.7], 'ravh9670': ['AH', 978.1, 991.8], 'ravh9830': ['AHHH', 994.1, 1011.6], 'ravh9990': ['AH', 1010.2, 1024.0], 'ravh10140': ['AH', 1025.2, 1038.9], 'ravh10250': ['AHH', 1036.2, 1051.9], 'ravh10390': ['AHH', 1050.1, 1065.7], 'ravh10520': ['AAHHH', 1063.1, 1081.9], 'ravh10660': ['AHHHH', 1077.4, 1096.8], 'ravh10800': ['AHH', 1091.7, 1107.3], 'ravh11080': ['AHH', 1119.1, 1134.7], 'ravh11320': ['AHH', 1143.1, 1158.8], 'ravh11490': ['AHHHH', 1160.1, 1179.3], 'ravh11650': ['A', 1176.2, 1187.8]}

r2 = {'ravh40': ['AHHHHHHHHHH', 15.1, 44.6], 'ravh580': ['AHHHHHHHHHH', 69.1, 98.5], 'ravh1290': ['AHHHHHH', 140.1, 162.9], 'ravh1920': ['AHHHHHHHHHH', 203.1, 232.5], 'ravh2330': ['AHHHHHHHHHHH', 244.1, 275.3], 'ravh2750': ['AHHHH', 286.1, 305.4], 'ravh3350': ['AHHHHHHHHHHH', 346.1, 377.2], 'ravh3800': ['AHHHHHHHHHHH', 391.1, 422.1], 'ravh4290': ['AHHHHHHHHH', 440.1, 467.8], 'ravh4770': ['AHHHHHHHHHHH', 488.1, 519.1], 'ravh5220': ['AHHHHHHH', 533.1, 557.5], 'ravh5930': ['AHHHHHHHHHHH', 604.2, 635.2], 'ravh6500': ['AHHHHHHHHHH', 661.1, 690.5], 'ravh7070': ['AHHHHHHHHHHH', 718.1, 749.7], 'ravh7770': ['AHHHHHHHH', 788.2, 814.3], 'ravh8230': ['AHHHHHHHHHHH', 834.1, 865.1], 'ravh8640': ['AHHHHHHHHHHH', 875.1, 906.1], 'ravh9200': ['AHHHHHHH', 931.1, 955.5], 'ravh9670': ['AHHHHHHHH', 978.1, 1004.3], 'ravh10140': ['AHHHHHHH', 1025.1, 1049.6], 'ravh10660': ['AHHHHHHHHHHH', 1077.1, 1108.2], 'ravh11080': ['AHHHHHHHHHHH', 1119.1, 1150.2]}

# orgarnise by value to dataframe, columns: ['platoon_type', 'start_time', 'end_time']
# transfer "AH" to length('AH')
# caculate each type's average time consumes

import pandas as pd


def to_dataframe(source: dict) -> pd.DataFrame:
    rows = []
    for key, value in source.items():
        platoon_type, start_time, end_time = value
        rows.append(
            {
                "id": key,
                "platoon_type": platoon_type,
                "start_time": float(start_time),
                "end_time": float(end_time),
            }
        )
    return pd.DataFrame(rows)


def encode_platoon_length(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["platoon_len"] = out["platoon_type"].str.len()
    return out


def add_duration(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["duration"] = out["end_time"] - out["start_time"]
    return out


def average_time_by_type_length(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("platoon_len", as_index=False)["duration"]
        .mean()
        .rename(columns={"duration": "avg_duration"})
        .sort_values("platoon_len")
    )


# step 1\: build DataFrames
df1 = to_dataframe(r1)
df2 = to_dataframe(r2)

df_sum = pd.concat([df1, df2], axis=0, ignore_index=True)
print(df_sum)
#%%
# step 2\: convert platoon type to length
df_sum = encode_platoon_length(df_sum)

# step 3\: calculate each row time consumption
df_sum = add_duration(df_sum)

# step 4\: average time consumption by type length
avg1 = average_time_by_type_length(df_sum)

print("average duration by platoon length")
print(avg1.to_string(index=False, float_format="%.2f"))