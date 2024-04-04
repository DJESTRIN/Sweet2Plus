import json

def SaveObj(FullPath: str, CurrentObject):
    """ Saves custom objects to json file
    Inputs:
    FullPath -- A string containing the entire directory and filename.json where the object will be saved. 
    CurrentObject -- The object we would like to save

    Outputs:
    None .... File will be saved to specified location
    """
    with open(FullPath, 'w') as file:
        json.dump(CurrentObject, file)

def LoadObj(FullPath: str):
    """ Saves custom objects to json file
    Inputs:
    FullPath -- A string containing the entire directory and filename.json where the object was saved

    Outputs:
    The object you saved in that json file
    """
    with open(FullPath, 'r') as file:
        CurrentObject = json.load(file)
    
    return CurrentObject