import pandas as pd

def get_stim_location(stim_type):
    stim_type = stim_type.lower()
    location = "Unknown"
    locations = ["Distal" ,"Proximal", "Cord"]
    for loc in locations:
        if loc.lower() in stim_type:
            location = loc
            break
    return location

def get_stim_method(stim_type):
    stim_type = stim_type.lower()
    stimulus = "Unknown"
    if ("touch" in stim_type) or ("pinch" in stim_type):
        stimulus = "Mechanical"
    elif ("electrical" in stim_type) or ("hz" in stim_type) or ("one stimulation" in stim_type):
        stimulus = "Electrical"
    return stimulus

def read_octopus_xlsx(xls_path):
    df = pd.read_excel(xls_path) 
    df2 = df.copy()
    df = df.drop(index=0)
    df = df.rename(columns=df2.iloc[0])
    df['Stim Location'] = df['Stimulation Type'].apply(get_stim_location)
    df['Stim Method'] = df['Stimulation Type'].apply(get_stim_method)
    df['Stimulation Class'] = df['Stim Location'] + ' ' + df['Stim Method']
    df.reset_index(inplace=True)
    return df