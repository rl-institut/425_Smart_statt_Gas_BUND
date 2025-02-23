import pypsa
import matplotlib.pyplot as plt
import pandas as pd
from functools import reduce

pd.set_option('display.width', 1000000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', '{:.5f}'.format)



def getIndexDe (df):
  return df.index.str.startswith('DE0')

def getBus0De (df):
  return df['bus0'].str.startswith('DE0')

def getBus1De (df):
  return df['bus1'].str.startswith('DE0')

def getDeExport (df):
  return (df['bus0'].str.startswith('DE0')) & (~(df['bus1'].str.startswith('DE0')))

def getDeImport (df):
  return (~(df['bus0'].str.startswith('DE0'))) & (df['bus1'].str.startswith('DE0'))

def getDeInner (df):
  return (df['bus0'].str.startswith('DE0')) & (df['bus1'].str.startswith('DE0'))

def getTheCarrier (df , carrier):
  return (df['carrier'] == carrier)

def getIndexDeCarrier (df , carrier):
  return getIndexDe(df) & getTheCarrier(df, carrier)

def getIndexDeInnerCarrier (df , carrier):
  return getDeInner(df) & getTheCarrier(df, carrier)

def getIndexDeExportCarrier (df, carrier):
  return getDeExport(df) & getTheCarrier(df, carrier)

def getIndexDeImportCarrier (df, carrier):
  return getDeImport(df) & getTheCarrier(df, carrier)

def make_grouper(name, items):
  def process_grouper(n, c):
    for item in items:
      itemC = item[0]
      itemFun = item[1]
      if (c == itemC):
        df = n.df(c)
        return getIndexSeries(df,itemFun(df))
      return getEmptyIndex()
  if name != '':
    grouperMap[name] = process_grouper
  return process_grouper

def getDeIndexes (component,  carriers):
  # de_XXX_grouper = make_grouper('de_XXX_grouper',[getDeIndexes('Link', ['DC'])])
  # de_XXX_grouper(c, 'Link').tolist()
  # n.statistics.optimal_capacity(groupby=de_XXX_grouper)

  def getIndexes (df):
    series_list = []

    for carrier in carriers:
      series_list.append(getIndexDeCarrier(df, carrier))

    result = reduce(lambda x, y: x | y, series_list)
    return result

  return [component, getIndexes] 

def getDeImportIndexes (component,  carriers):
  def getIndexes (df):
    series_list = []

    for carrier in carriers:
      series_list.append(getIndexDeImportCarrier(df, carrier))

    result = reduce(lambda x, y: x | y, series_list)
    return result

  return [component, getIndexes] 

def getDeExportIndexes (component,  carriers):
  def getIndexes (df):
    series_list = []

    for carrier in carriers:
      series_list.append(getIndexDeExportCarrier(df, carrier))

    result = reduce(lambda x, y: x | y, series_list)
    return result

  return [component, getIndexes] 


def getNByGroup (n, component, grouper):
  indexes =  grouper(n, component)
  dict = {
    'Generator': n.generators,
    'Link': n.links,
    'Line': n.lines,
    'Store': n.stores,
    'StorageUnit': n.storage_units,
    'Load': n.loads
  }

  return dict[component].loc[indexes]

def getNTimePropByGroup (n, component, prop, grouper):
  indexes =  grouper(n, component)
  dict = {
    'Generator': n.generators_t,
    'Link': n.links_t,
    'Line': n.lines_t,
    'Store': n.stores_t,
    'StorageUnit': n.storage_units_t,
    'Load': n.loads_t
  }

  return dict[component][prop][indexes]




def getIndexSeries(df, condition):
  return df[condition].index.to_series()

def getEmptyIndex ():
  return pd.Index([]).to_series()

def replaceCarrier (df, old, new): 
  df['carrier'] = df['carrier'].replace(old, new)
  return df

grouperMap = {}

def de_grouper(n, c):
  if (c == 'Line'):
    dfLine = n.df(c)
    indexsLine = dfLine[dfLine['bus1'].str.startswith('DE0')].index.to_series()
    return indexsLine

  if (c == 'Link'):
    dfLink = n.df(c)
    indexsLink = dfLink[(dfLink['bus1'].str.startswith('DE0')) 
                        | ((dfLink['bus0'].str.startswith('DE0') & dfLink['bus1'].str.startswith('EU')))
                      ].index.to_series()
    return indexsLink

  if (c == 'Generator'):
    dfLink = n.df(c)
    indexsLink = dfLink[dfLink['bus'].str.startswith('DE0')]

  if (c == 'Store'):
    dfLink = n.df(c)
    indexsLink = dfLink[dfLink['bus'].str.startswith('DE0')]

  if (c == 'StorageUnit'):
    dfLink = n.df(c)
    indexsLink = dfLink[dfLink['bus'].str.startswith('DE0')]
  
  df = n.df(c)
  return df[df.index.str.startswith('DE0')].index.to_series()
grouperMap['de_grouper'] = de_grouper


def de_import_elec_grouper(n, c):
  if (c == 'Line'):
    dfLine = n.df(c)
    return dfLine[dfLine['bus1'].str.startswith('DE0') 
                  & (~(dfLine['bus0'].str.startswith('DE0')))].index.to_series()
  
  if (c == 'Link'):
    dfLink = n.df(c)
    return dfLink[(dfLink['bus1'].str.startswith('DE0')) 
                  & (dfLink['carrier'] == 'DC') 
                  & (~(dfLink['bus0'].str.startswith('DE0')))].index.to_series()

  return pd.Index([]).to_series()
grouperMap['de_import_elec_grouper'] = de_import_elec_grouper

def de_import_elec_reverse_grouper(n, c):
  if (c == 'Line'):
    dfLink = n.df(c)
    return dfLink[(~(dfLink['bus1'].str.startswith('DE0'))) 

                  & (dfLink['bus0'].str.startswith('DE0'))].index.to_series()

  return pd.Index([]).to_series()
grouperMap['de_import_elec_reverse_grouper'] = de_import_elec_reverse_grouper

def de_export_elec_grouper(n,c):
  if (c == 'Line'):
    dfLine = n.df(c)
    return dfLine[(dfLine['bus0'].str.startswith('DE0')) 
                  & (~(dfLine['bus1'].str.startswith('DE0'))) ].index.to_series()
  
  if (c == 'Link'):
    dfLink = n.df(c)
    return dfLink[(dfLink['bus0'].str.startswith('DE0')) 
                  & (dfLink['carrier'] == 'DC')  
                  & (~(dfLink['bus1'].str.startswith('DE0')))].index.to_series()

  return pd.Index([]).to_series()
grouperMap['de_export_elec_grouper'] = de_export_elec_grouper


## section generator
de_pv_generator_grouper = make_grouper('de_pv_generator_grouper', [
  getDeIndexes('Generator', ['solar', 'solar rooftop', 'solar-hsat'])
])
de_wind_generator_grouper = make_grouper('de_wind_generator_grouper', [
  getDeIndexes('Generator', ['offwind-ac', 'offwind-dc', 'ffwind-float', 'onwind'])
])

de_ror_generator_grouper = make_grouper('de_ror_generator_grouper', [
  getDeIndexes('Generator', ['ror'])
])

de_co2_generator_grouper = make_grouper('de_co2_generator_grouper', [
   getDeIndexes('Link', ['OCGT','OCCT', 'urban central CHP'])
])

de_biomass_generator_grouper = make_grouper('de_co2_generator_grouper', [
   getDeIndexes('Link', ['urban central solid biomass CHP'])
])

de_geothermal_generator_grouper = make_grouper('de_geothermal_generator_grouper', [
  getDeIndexes('Link', ['geothermal organic rankine cycle'])
])

def de_all_generator_grouper(n,c):
  if ( c == 'Generator'):
    df = n.df(c)
    index =  df[ 
              ((df.index.str.startswith('DE0')) & (df['carrier'] == 'offwind-ac')) 
              | ((df.index.str.startswith('DE0')) & (df['carrier'] == 'offwind-dc')) 
              | ((df.index.str.startswith('DE0')) & (df['carrier'] == 'offwind-float')) 
              | ((df.index.str.startswith('DE0')) & (df['carrier'] == 'onwind')) 
              | ((df.index.str.startswith('DE0')) & (df['carrier'] == 'ror')) 
              | ((df.index.str.startswith('DE0')) & (df['carrier'] == 'solar')) 
              | ((df.index.str.startswith('DE0')) & (df['carrier'] == 'solar rooftop')) 
              | ((df.index.str.startswith('DE0')) & (df['carrier'] == 'solar-hsat'))
            ].index.to_series()
    return index
  
  if ( c == 'Link'):
    df = n.df(c)
    index =  df[((df['bus0'].str.startswith('DE0')) & (df['carrier'] == 'OCGT')) |
              ((df['bus0'].str.startswith('DE0')) & (df['carrier'] == 'H2 Fuel Cell')) |
              ((df['bus0'].str.startswith('DE0')) & (df['carrier'] == 'H2 turbine')) |
              ((df['bus0'].str.startswith('DE0')) & (df['carrier'] == 'urban central solid biomass CHP')) |
             ((df['bus0'].str.startswith('DE0')) & (df['carrier'] == 'urban central CHP'))   |
              ((df['bus0'].str.startswith('DE0')) & (df['carrier'] == 'geothermal organic rankine cycle')) 
              ].index.to_series()
    return index
  return getEmptyIndex()
grouperMap['de_all_generator_grouper'] = de_all_generator_grouper


def de_generator_grouper(n,c):
  if ( c == 'Generator'):
    df = n.df(c)
    index =  df[
              ((df.index.str.startswith('DE0')) & (df['carrier'] == 'offwind-ac')) 
              | ((df.index.str.startswith('DE0')) & (df['carrier'] == 'offwind-dc')) 
              | ((df.index.str.startswith('DE0')) & (df['carrier'] == 'offwind-float')) 
              | ((df.index.str.startswith('DE0')) & (df['carrier'] == 'onwind')) 
              | ((df.index.str.startswith('DE0')) & (df['carrier'] == 'ror')) 
              | ((df.index.str.startswith('DE0')) & (df['carrier'] == 'solar')) 
              | ((df.index.str.startswith('DE0')) & (df['carrier'] == 'solar rooftop')) 
              | ((df.index.str.startswith('DE0')) & (df['carrier'] == 'solar-hsat'))
            ].index.to_series()
    return index
  
  if ( c == 'Link'):
    df = n.df(c)
    index =  df[((df['bus0'].str.startswith('DE0')) & (df['carrier'] == 'OCGT')) |
              ((df['bus0'].str.startswith('DE0')) & (df['carrier'] == 'H2 Fuel Cell')) |
              ((df['bus0'].str.startswith('DE0')) & (df['carrier'] == 'H2 turbine')) |
              ((df['bus0'].str.startswith('DE0')) & (df['carrier'] == 'urban central solid biomass CHP')) |
             ((df['bus0'].str.startswith('DE0')) & (df['carrier'] == 'urban central CHP'))  |
              ((df['bus0'].str.startswith('DE0')) & (df['carrier'] == 'geothermal organic rankine cycle')) 
              ].index.to_series()
    return index
  return getEmptyIndex()
grouperMap['de_generator_grouper'] = de_generator_grouper


def de_heat_generator_grouper(n,c):
  if (c =='Generator'):
    df = n.df(c)
    return getIndexSeries(df, getIndexDeCarrier(df, 'rural solar thermal') |
             getIndexDeCarrier(df, 'urban central solar thermal') |
             getIndexDeCarrier(df, 'urban decentral solar thermal') 
            )

  if (c == 'Link'):
    df = n.df(c)
    return getIndexSeries(df, 
                      getIndexDeCarrier(df, 'urban decentral gas boiler') |
                        getIndexDeCarrier(df, 'urban decentral biomass boiler') |
                        getIndexDeCarrier(df, 'urban decentral air heat pump') |
                        getIndexDeCarrier(df, 'urban decentral resistive heater') |
                        getIndexDeCarrier(df, 'urban central solid biomass CHP') |
                        getIndexDeCarrier(df, 'urban central resistive heater') |
                        getIndexDeCarrier(df, 'urban central gas boiler') |
                        getIndexDeCarrier(df, 'urban central air heat pump') |
                        getIndexDeCarrier(df, 'urban central CHP') |
                        getIndexDeCarrier(df, 'rural resistive heater') |
                        getIndexDeCarrier(df, 'rural ground heat pump') |
                        getIndexDeCarrier(df, 'rural gas boiler') |
                        getIndexDeCarrier(df, 'rural biomass boiler') |
                        getIndexDeCarrier(df, 'rural air heat pump') |
                        getIndexDeCarrier(df, 'geothermal heat') |
                        getIndexDeCarrier(df, 'geothermal district heat') 
                      )
  return getEmptyIndex()
grouperMap['de_heat_generator_grouper'] = de_heat_generator_grouper

def de_biogas_generator_grouper(n, c):
  if (c =='Generator'):
    df = n.df(c)
    return getIndexSeries(df, getIndexDeCarrier(df, 'biogas'))
  return getEmptyIndex()
grouperMap['de_biogas_generator_grouper'] = de_biogas_generator_grouper


# gas start
de_gas_generator = make_grouper('de_gas_generator', [
  getDeIndexes('Generator', ['gas'])
])
de_sabatier = make_grouper('de_sabatier_generator', [
  getDeIndexes('Link', ['Sabatier'])
])
de_biogas_to_gas = make_grouper('', [
  getDeIndexes('Link', ['biogas to gas'])
])
de_gas_pipeline_import = make_grouper('de_sabatier_link', [
  getDeImportIndexes('Link', ['gas pipeline'])
])
de_gas_use = make_grouper('de_gas_use', [
  getDeIndexes('Link', ['OCGT', 'SMR CC', 'SMR', 'urban central gas boiler','urban central CHP', 'urban central CHP CC','gas for industry', 'gas for industry CC', 'rural gas boiler','urban decentral gas boiler'])
])
de_gas_use_elc = make_grouper('de_gas_use', [
  getDeIndexes('Link', ['OCGT','urban central CHP CC',])
])

de_gas_use_warm = make_grouper('de_gas_use', [
  getDeIndexes('Link', [ 'urban central gas boiler','urban central CHP', 'urban central CHP CC', 'rural gas boiler','urban decentral gas boiler'])
])

de_gas_use_SMR = make_grouper('de_gas_use', [
  getDeIndexes('Link', ['SMR CC', 'SMR', 'gas for industry', 'gas for industry CC'])
])
# gas end

# biomass start
de_biomass_chp =  make_grouper('de_biomass_chp', [
  getDeIndexes('Link', ['urban central solid biomass CHP'])
])
# biomass end


# battery start
de_battery_store =  make_grouper('de_battery_store', [
  getDeIndexes('Store', ['battery', "home battery"])
])
de_home_battery_store =  make_grouper('de_home_battery_store', [
  getDeIndexes('Store', ['home battery'])
])

de_battery_discharger = make_grouper('de_battery_discharger', [
  getDeIndexes('Link', ['battery charger', 'home battery discharger'])
])

de_home_battery_discharger = make_grouper('de_home_battery_discharger', [
  getDeIndexes('Link', ['home battery discharger'])
])

# battery end

de_gas_pipeline_export = make_grouper('de_gas_pipeline_export', [
  getDeExportIndexes('Link', ['gas pipeline'])
])


def de_gas_generator_grouper(n,c):
  # if (c =='Generator'):
  #   df = n.df(c)
  #   return getIndexSeries(df, getIndexDeCarrier(df, 'gas'))
  
  if (c == 'Link'):
    df = n.df(c)
    return getIndexSeries(df, 
                      getIndexDeCarrier(df, 'Sabatier') 
                      # getIndexDeCarrier(df, 'biogas to gas')# |
                      # getIndexDeImportCarrier(df, 'gas pipeline') 
                      )
  return getEmptyIndex()

grouperMap['de_gas_generator_grouper'] = de_gas_generator_grouper

def de_gas_use_grouper(n,c):
  if (c == 'Link'):
    df = n.df(c)
    return getIndexSeries(df, 
                      getIndexDeCarrier(df, 'OCGT') |
                      getIndexDeCarrier(df, 'SMR') |
                      getIndexDeCarrier(df, 'urban central gas boiler') |
                      getIndexDeCarrier(df, 'urban central CHP') |
                      getIndexDeCarrier(df, 'urban central CHP') |
                      getIndexDeCarrier(df, 'rural gas boiler') |
                      getIndexDeCarrier(df, 'urban decentral gas boiler') 
                      )
  return getEmptyIndex()
grouperMap['de_gas_use_grouper'] = de_gas_use_grouper

def de_phs_grouper(n, c):
  if (c == "StorageUnit"):
    df = n.df(c)
    return getIndexSeries(df, 
                      getIndexDeCarrier(df, 'PHS')
                      )
  return getEmptyIndex()
grouperMap['de_phs_grouper'] = de_phs_grouper

def de_elec_store_grouper(n, c):
  if (c == "StorageUnit"):
    df = n.df(c)
    return getIndexSeries(df, 
                      getIndexDeCarrier(df, 'PHS') | 
                      getIndexDeCarrier(df, 'hydro')
                      )
  if (c == 'Store'):

    df = n.df(c)
    return getIndexSeries(df,
                         getIndexDeCarrier(df, 'EV battery')
                         | getIndexDeCarrier(df, 'battery') 
                         | getIndexDeCarrier(df, 'home battery')
                        )
                        
  return getEmptyIndex()
grouperMap['de_elec_store_grouper'] = de_elec_store_grouper

def de_battery_store_grouper(n, c):
  if (c == 'Store'):
    df = n.df(c)
    return getIndexSeries(df,
                         getIndexDeCarrier(df, 'battery') | 
                         getIndexDeCarrier(df, 'home battery')
                        )
  return getEmptyIndex()
grouperMap['de_battery_store_grouper'] = de_battery_store_grouper
 

def de_h2_to_elc_gropuer(n,c):
  if ( c == 'Link'):
    df = n.df(c)
    return getIndexSeries(df,
                         getIndexDeCarrier(df, 'H2 Fuel Cell') 
                        )
  return getEmptyIndex()
grouperMap['de_h2_to_elc_gropuer'] = de_h2_to_elc_gropuer

def de_elc_to_h2_grouper(n,c):
  if ( c == 'Link'):
    df = n.df(c)
    return getIndexSeries(df,
                         getIndexDeCarrier(df, 'H2 Electrolysis') 
                        )
  return getEmptyIndex()
grouperMap['de_elc_to_h2_grouper'] = de_elc_to_h2_grouper


def de_elec_use_grouper(n,c):
  if (c == 'Load'):
    df = n.df(c)
    return getIndexSeries(df,
                          getIndexDeCarrier(df, 'agriculture electricity') |
                          getIndexDeCarrier(df, 'electricity') |
                          getIndexDeCarrier(df, 'industry electricity')  |
                          getIndexDeCarrier(df, 'land transport EV')  
                          )
  if (c == 'Link'):
    
    df = n.df(c)
    return getIndexSeries(df,
                          getIndexDeCarrier(df, 'urban central air heat pump') |
                          getIndexDeCarrier(df, 'urban central resistive heater') |
                          getIndexDeCarrier(df, 'rural air heat pump') |
                          getIndexDeCarrier(df, 'rural ground heat pump') |
                          getIndexDeCarrier(df, 'rural resistive heater') |
                          getIndexDeCarrier(df, 'urban decentral air heat pump') |
                          getIndexDeCarrier(df, 'urban decentral resistive heater') |
                          getIndexDeCarrier(df, 'urban decentral resistive heater') |
                          getIndexDeCarrier(df, 'DAC')
                          )
  return getEmptyIndex()
grouperMap['de_elec_use_grouper'] = de_elec_use_grouper

def de_warm_link(n, c):
  if (c == 'Link'):
  
    df = n.df(c)
    return getIndexSeries(df,
                          getIndexDeCarrier(df, 'urban central air heat pump') |
                          getIndexDeCarrier(df, 'urban central resistive heater') |
                          getIndexDeCarrier(df, 'rural air heat pump') |
                          getIndexDeCarrier(df, 'rural ground heat pump') |
                          getIndexDeCarrier(df, 'rural resistive heater') |
                          getIndexDeCarrier(df, 'urban decentral air heat pump') |
                          getIndexDeCarrier(df, 'urban decentral resistive heater') |
                          getIndexDeCarrier(df, 'DAC')
                          )
  return getEmptyIndex()
grouperMap['de_warm_link'] = de_warm_link



de_elec_distrbution_use_grouper = make_grouper('de_elec_distrbution_use_grouper', [
    getDeIndexes('Link', ['electricity distribution grid'])
  ])

de_elec_methanol_use = make_grouper('de_elec_methanol_use', [
  getDeIndexes('Link', ['methanolisation'])
])


def de_elec_charge_grouper(n, c):
  if (c == 'Link'):
    df = n.df(c)
    return getIndexSeries(df,
                          getIndexDeCarrier(df, 'BEV charger') |
                          getIndexDeCarrier(df, 'urban central air heat pump') |
                          getIndexDeCarrier(df, 'urban central resistive heater') |
                          getIndexDeCarrier(df, 'urban central resistive heater')    
                          )
  return getEmptyIndex()
grouperMap['de_elec_charge_grouper'] = de_elec_charge_grouper

def de_dsm_grouper(n, c):
  if (c == 'Link'):
    df = n.df(c)
    return getIndexSeries(df,
      getIndexDeCarrier(df, 'BEV charger')
      )
  if (c == 'Link'):
    df = n.df(c)
    return getIndexSeries(df,
      getIndexDeCarrier(df, 'EV battery')
      )
  return getEmptyIndex()
grouperMap['de_dsm_grouper'] = de_dsm_grouper

# secthion bev grouper
de_v2g = make_grouper('de_v2g', [getDeIndexes('Link', ['V2G'])])
de_ev_battery = make_grouper('de_ev_battery', [getDeIndexes('Store', ['EV battery'])])


# section h2 grouper
de_h2_FC_grouper = make_grouper('de_h2_FC_grouper',[('Link', lambda df: 
                       getIndexDeCarrier(df, 'H2 Fuel Cell'))])
de_h2_turbine_grouper = make_grouper('de_h2_turbine_grouper',[('Link', lambda df: 
                       getIndexDeCarrier(df, 'H2 turbine'))])
de_h2_dispatch_grouper = make_grouper('de_h2_d_grouper',[
  getDeIndexes('Link', ['H2 Fuel Cell', 'H2 turbine'])])
de_h2_store_grouper = make_grouper('de_h2_store_grouper',[('Store', lambda df: 
                       getIndexDeCarrier(df, 'H2 Store'))])
de_h2_Electrolysis_grouper = make_grouper('de_h2_Electrolysis_grouper',[('Link', lambda df: 
                       getIndexDeCarrier(df, 'H2 Electrolysis'))])

de_h2_SMR_grouper = make_grouper('de_h2_SMR_grouper', [getDeIndexes('Link', ['SMR', 'SMR CC'])])

de_h2_use_grouper = make_grouper('de_h2_load_grouper',[
  #  getDeIndexes('Load', ['H2 for industry']),
   getDeIndexes('Link', ['Sabatier', 'methanolisation']),
   ])

de_h2_FT_grouper = make_grouper('de_h2_FT_grouper',[
  #  getDeIndexes('Load', ['H2 for industry']),
   getDeIndexes('Link', ['Fischer-Tropsch']),
   ])

de_h2_industry_grouper = make_grouper('de_h2_industry_grouper',[
    getDeIndexes('Load', ['H2 for industry']),
   ])
de_import_h2_grouper = make_grouper('de_import_h2_grouper', [
  getDeImportIndexes('Link', ['H2 pipeline'])
  ])
de_export_h2_grouper = make_grouper('de_export_h2_grouper', [
  getDeExportIndexes('Link', ['H2 pipeline'])
  ])

# section ror
de_ror_grouper = make_grouper('de_ror_grouper', [
  getDeImportIndexes('Generator', ['ror'])
  ])

# sector biomass
de_solid_biomass = make_grouper('de_solid_biomass', [
  getDeIndexes('Generator', ['solid biomass'])
])

de_solid_biomass = make_grouper('de_solid_biomass', [
  getDeIndexes('Generator', ['solid biomass'])
])

de_solid_biomass_chp = make_grouper('de_solid_biomass_chp', [
  getDeIndexes('Link', ['urban central solid biomass CHP', 'urban central solid biomass CHP CC'])
])

de_solid_biomass_industry = make_grouper('de_solid_biomass_industry', [
  getDeIndexes('Link', ['solid biomass for industry', 'solid biomass for industry CC'])
])

de_rural_biomass_boiler = make_grouper('de_rural_biomass_boiler', [
    getDeIndexes('Link', ['rural biomass boiler'])
])

de_urban_decentral_biomass_boiler = make_grouper('de_rural_biomass_boiler', [
    getDeIndexes('Link', ['urban decentral biomass boiler'])
])





def eu_grouper(n, c):
  df = n.df(c)
  print('----------------',c)
  print(df.groupby('carrier').size())
  indexs = df[df.index.astype(str).str.contains('EU')].index.to_series()
  return indexs
grouperMap['eu_grouper'] = eu_grouper

def dataframe_to_table(df):
    df = df.fillna('/')
    result_list = []
    for index, row in df.iterrows():
        formatted_elements = []
        formatted_elements.append(f'[{index}]')
        for item in row:
            if isinstance(item, float):
                formatted_number = f'{item:.3f}'.replace('.', ',')
                formatted_elements.append(f'[{formatted_number}]') 
            else:
                formatted_elements.append(f'[{item}]')
        formatted_row = ', '.join(formatted_elements)
        result_list.append(f'  {formatted_row},') 
    return '\n'.join(result_list)

def assignGrouper (pypsa):
  for kye, value in grouperMap.items():
    pypsa.statistics.groupers.add_grouper(kye, value)
assignGrouper(pypsa)


def renameHeatCarrier (df):
  df.rename(index={'geothermal district heat': 'Geothermie'}, inplace=True)
  df.rename(index={'geothermal heat': 'Geothermie'}, inplace=True)
  df.rename(index={'rural air heat pump': 'Strom zu Wärme'}, inplace=True)
  df.rename(index={'rural biomass boiler': 'Biomassekessel'}, inplace=True)
  df.rename(index={'rural gas boiler': 'Gaskessel'}, inplace=True)
  df.rename(index={'rural ground heat pump': 'Strom zu Wärme'}, inplace=True)
  df.rename(index={'rural resistive heater': 'Strom zu Wärme'}, inplace=True)
  df.rename(index={'rural resistive heater': 'Solarthermie'}, inplace=True)
  df.rename(index={'rural solar thermal': 'Solarthermie'}, inplace=True)
  df.rename(index={'urban central CHP': 'Gas CHP'}, inplace=True)
  df.rename(index={'urban central air heat pump': 'Strom zu Wärme'}, inplace=True)
  df.rename(index={'urban central gas boiler': 'Gaskessel'}, inplace=True)
  df.rename(index={'urban central resistive heater': 'Strom zu Wärme'}, inplace=True)
  df.rename(index={'urban central solar thermal': 'Solarthermie'}, inplace=True)
  df.rename(index={'urban central solid biomass CHP': 'Biomasse CHP'}, inplace=True)
  df.rename(index={'urban decentral air heat pump': 'Strom zu Wärme'}, inplace=True)
  df.rename(index={'urban decentral biomass boiler': 'Biomassekessel'}, inplace=True)
  df.rename(index={'urban decentral gas boiler': 'Gaskessel'}, inplace=True)
  df.rename(index={'urban decentral resistive heater': 'Strom zu Wärme'}, inplace=True)
  df.rename(index={'urban decentral solar thermal': 'Solarthermie'}, inplace=True)

  return df.groupby('carrier').sum()

def getDiffTableColum (cb, scenarios):
  result = {}
  for scenario in scenarios:
    result[scenario.name] = cb(scenario.n)
  
  return result