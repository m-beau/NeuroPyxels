
from statsmodels.nonparametric.smoothers_lowess import lowess


def smooth(arr, frac=0.06, it=0, frac2=0.005, spec='acg', cbin=0.1, cwin=100, lms=3):
    '''cbin and cwin in milliseconds.
    frac is the fraction looked ahead for smoothing on the sides on the array, frac2 between -lms and + lms in milliseconds.'''
    arr=np.asarray(arr)
    if spec=='acg' or spec=='ccg': # smooth the central bins with a low frac, sides with high frac
        halfcenter= (lms/cbin)# Left smoothing difference limit, in milliseconds
        l = int((cwin/cbin)*1./2 - halfcenter); r = int(l+halfcenter*2);
        sarr1=lowess(arr, np.arange(len(arr)), is_sorted=True, frac=frac, it=it)[:,1].flatten()[:l]
        sarr2=lowess(arr, np.arange(len(arr)), is_sorted=True, frac=frac2, it=it)[:,1].flatten()[l:r]
        sarr3=lowess(arr, np.arange(len(arr)), is_sorted=True, frac=frac, it=it)[:,1].flatten()[r:]
        sarr = np.append(np.append(sarr1, sarr2), sarr3)
    else:
        sarr=lowess(arr, np.arange(len(arr)), is_sorted=True, frac=frac, it=it)[:,1].flatten()
    return sarr
