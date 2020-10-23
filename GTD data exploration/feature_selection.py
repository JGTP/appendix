# filter_columns = ['iyear', 'imonth', 'iday', 'country', 'region', 'provstate', 'city', 'latitude', 'longitude', 'gname',\
# 'success', 'suicide', 'attacktype1', 'targtype1',  'individual', 'nperps', 'nperpcap', 'claimed', 'claimmode', 'compclaim', \
# 'weaptype1', 'weapsubtype1', 'nkill', 'nkillus', 'nkillter', 'nwound', 'nwoundus', 'nwoundte', 'property', 'propextent', \
# 'propvalue', 'ishostkid', 'nhostkid', 'nhostkidus', 'nhours', 'ndays', 'ransom', 'ransomamt', \
# 'ransomamtus', 'ransompaid', 'ransompaidus', 'hostkidoutcome', 'nreleased']

exclude_columns = ['summary', 'related', 'provstate', 'city', 'location', 'weapdetail', 'gsubname', 'gname2', 'gsubname2',
                   'gname3', 'gsubname3', 'motive', 'propcomment', 'divert', 'kidhijcountry', 'ransomnote', 'addnotes', 'scite1', 'scite2', 'scite3', 'dbsource', 'approxdate', 'resolution', 'circumstance_time']

date_cols = ['iyear', 'imonth', 'iday', ]

geo_cols = ['country_txt',  'region_txt', ]

coordinates = ['longitude', 'latitude']

minus_9_is_unknown = ['compclaim', 'property', 'ishostkid', 'ransom', ]
# minus_9_is_unknown = [ 'doubtterr','INT_LOG', 'INT_IDEO', 'INT_MISC', 'INT_ANY']

minus_99_is_unknown = ['nperps', 'nperpcap', 'nhostkid', 'nhours', 'ransomamt',
                       'ransomamtus', 'ransompaid', 'ransompaidus', 'nreleased', 'propvalue', 'ndays']

numerical_cols = ['nperps', 'nperpcap', 'nkill', 'nkillus', 'nkillter', 'nwound', 'nwoundus', 'nwoundte', 'propvalue', 'nhostkid', 'nhostkidus', 'nhours', 'ndays', 'ransomamt',
                  'ransomamtus', 'ransompaid', 'ransompaidus', 'nreleased']

# categorical_string_cols = []

categorical_string_cols = []

excluded_categorical_string_cols = ['attacktype1_txt', 'attacktype2_txt', 'attacktype3_txt', 'weaptype1_txt', 'weaptype2_txt', 'weaptype3_txt', 'weaptype4_txt',
                                    'weapsubtype1_txt', 'weapsubtype2_txt', 'weapsubtype3_txt', 'weapsubtype4_txt', 'corp1', 'corp2', 'corp3', 'target1', 'target2', 'target3', 'natlty1_txt', 'natlty2_txt', 'natlty3_txt',
                                    'alternative_txt',  'targtype1_txt', 'targsubtype1_txt', 'targtype2_txt', 'targsubtype2_txt', 'targtype3_txt',
                                    'targsubtype3_txt', 'propextent_txt', 'hostkidoutcome_txt', 'claimmode_txt', 'claimmode2_txt', 'claimmode3_txt']

categorical_number_columns = ['extended', 'attacktype1', 'attacktype2', 'attacktype3', 'weaptype1',  'weaptype2', 'weaptype3', 'weaptype4',
                              'weapsubtype1', 'weapsubtype2', 'weapsubtype3', 'weapsubtype4',   'success', 'suicide',
                              'claimed', 'claimmode', 'claim2', 'claimmode2', 'claim3', 'claimmode3', 'compclaim', 'property', 'ishostkid', 'ransom',  'targtype1', 'targsubtype1',
                              'targtype2', 'targsubtype2',   'targtype3', 'targsubtype3', 'propextent', 'hostkidoutcome', 'multiple', ]

exclude_categorical_number_columns = ['crit1', 'crit2', 'crit3', 'vicinity',  'individual', 'specificity', 'doubtterr', 'alternative', 'country',
                                      'region',  'natlty1',  'natlty2',  'natlty3', 'guncertain1', 'guncertain2', 'guncertain3', 'INT_LOG', 'INT_IDEO', 'INT_MISC', 'INT_ANY', ]

text_cols = ['gname']
