import json

def readJson2Dict(file_name):
    f = open(file_name, "r")
    out = json.load(f)
    f.close()
    return out


def writeJsonFromDict(file_name, write_dict):
    f = open(file_name, "w")
    json.dump(write_dict, f)
    f.close()