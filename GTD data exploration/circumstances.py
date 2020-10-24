import pandas as pd


def construct_time(row):
    return str(row['iyear']) + '-' + str(row['imonth']) +\
        '-' + str(row['iday'])


def construct_nvictims(row):
    nkill = determine_number(row['nkill'])
    nwound = determine_number(row['nwound'])
    nkillter = determine_number(row['nkillter'])
    nwoundter = determine_number(row['nwoundte'])
    return nkill - nkillter + nwound - nwoundter


def determine_number(cell):
    number = pd.to_numeric(cell, errors='coerce')
    if number is None:
        return 0
    if number < 0:
        return 0
    try:
        number = int(number)
    except ValueError:
        return 0
    return number


def construct_location(row):
    if row['city'] != None:
        if row['location'] != None:
            location = str(row['city']) + '-' + str(row['location'])
        else:
            location = row['city']
    elif row['location'] != None:
        location = row['location']
    else:
        return None
    location = str(location).replace('"', '')
    location = str(location).replace('(', '_')
    location = str(location).replace(')', '_')
    location = str(location).replace(',', '.')
    location = str(location).replace(':', '--')
    target = str(target).replace('#', '--')
    target = str(target).replace(' ', ' ')  # thin space
    target = str(target).replace('  ', '.')  # double space
    return location


def construct_target(row):
    if row['target1'] != None:
        target = str(row['target1'])
        if row['corp1'] != None:
            target = target + '-' + str(row['corp1'])
    if target != None:
        target = str(target).replace('"', '')
        target = str(target).replace('(', '_')
        target = str(target).replace(')', '_')
        target = str(target).replace(',', '.')
        target = str(target).replace(':', '--')
        target = str(target).replace('#', '--')
        target = str(target).replace(' ', ' ')  # thin space
        target = str(target).replace('  ', '.')  # double space
    return target
