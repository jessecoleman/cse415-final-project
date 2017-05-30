import NaiveBayesClassifier as nbc
import pandas as pd
import numpy as np

if __name__ == '__main__':
    facts = pd.read_csv('2016-us-election/county_facts.csv',sep=',')
    facts_features = pd.read_csv('2016-us-election/county_facts_dictionary.csv',sep=',')
    results = pd.read_csv('2016-us-election/primary_results.csv',sep=',')
    dem = results[results.party=='Democrat']
    m = dem.groupby('fips')['fraction_votes'].transform(max) \
            == dem['fraction_votes']
    dem_results = pd.merge(facts, dem[m][['fips','candidate']], how='inner', on='fips')
    dem_results = dem_results.sample(frac=1)
    rep = results[results.party=='Republican']
    m = rep.groupby('fips')['fraction_votes'].transform(max) \
            == rep['fraction_votes']
    rep_results = pd.merge(facts, rep[m][['fips','candidate']] , how='inner', on='fips')
    rep_results = rep_results.sample(frac=1)
    d_data = dem_results.ix[1:,3:dem_results.shape[1]-1]
    d_labels = dem_results.ix[1:,dem_results.shape[1]-1]
    r_data = rep_results.ix[1:,3:rep_results.shape[1]-1]
    r_labels = rep_results.ix[1:,rep_results.shape[1]-1]
    nbc.REPORTING = False
    nbc.k_fold(np.array(d_data), np.array(d_labels),4)
    nbc.k_fold(np.array(r_data), np.array(r_labels),4)
    
