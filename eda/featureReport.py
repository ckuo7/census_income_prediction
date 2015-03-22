__author__ = 'chi-liangkuo'

import pandas as pd
from collections import Counter

def featureReport():

    df = pd.read_csv('../data/census-income.data',header=None)
    df2 = pd.read_csv('../data/census-income.test',header =None)

    cols = ["age","class of worker","detailed industry recode","detailed occupation recode","education","wage per hour","enroll in edu inst last wk","marital stat",
            "major industry code","major occupation code","race","hispanic origin","sex","member of a labor union","reason for unemployment","full or part time employment stat",
            "capital gains","capital losses","dividends from stocks","tax filer stat","region of previous residence","state of previous residence","detailed household and family stat",
            "detailed household summary in household","instance weight","migration code-change in msa","migration code-change in reg","migration code-move within reg","live in this house 1 year ago","migration prev res in sunbelt",
            "num persons worked for employer","family members under 18","country of birth father","country of birth mother","country of birth self","citizenship","own business or self employed",
            "fill inc questionnaire for veteran's admin","veterans benefits","weeks worked in year","year","income level"]

    count = 0
    print len(cols)
    df.columns = cols
    df2.columns = cols
    nrow = df.values.shape[0]
    for i in df.columns:

        print "=============",i,"   :",count," feature type: ",df[i].dtype.name,"=================="
        #if df[i].dtype.name == 'object':
        print "key length: ",len(Counter(df[i].values).keys())

        for k,v in Counter(df[i].values).items():
            if i is not "instance weight":
                print "%2.4f " % round(v/float(nrow)*100,4),"   ",k

        print "************* test *************"
        print "key length: ",len(Counter(df2[i].values).keys())
        for k,v in Counter(df2[i].values).items():

            if i is not "instance weight":
                print "%2.4f " % round(v/float(nrow)*100,4),"   ",k

        #else:
        #    print df[i].describe()
        print "\n"
        count += 1


if __name__ == "__main__":


    featureReport()
