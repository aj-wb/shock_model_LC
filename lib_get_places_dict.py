import pandas as pd
def get_places_dict(myC,reverse=False):
    """Get economy-level names of provinces or districts (p-code)
    and larger regions if available (r-code)
    Parameters
    ----------
    myC : str
        ISO
    Returns
    -------
    p_code : Series
        Series, province/district code as index, province/district names as values.
    r_code : Series
        region code as index, region names as values.
    """

    p_code,r_code = None,None

    if myC == 'PH':
        p_code = pd.read_excel('in/FIES_provinces.xlsx')[['province_code','province_AIR']].set_index('province_code').squeeze()
        #p_code[97] = 'Zamboanga del Norte'
        #p_code[98] = 'Zamboanga Sibugay'
        if reverse: p_code = p_code.reset_index().set_index('province_AIR')

        r_code = pd.read_excel('in/FIES_regions.xlsx')[['region_code','region_name']].set_index('region_code').squeeze()
        if reverse: r_code = r_code.reset_index().set_index('region_name')

    try: p_code = p_code.to_dict()
    except: pass
    try: r_code = r_code.to_dict()
    except: pass

    return p_code,r_code