def limit(oa,size,sizea):
    bia = (size-max(oa)+min(oa))/2
    if sizea-max(oa) > min(oa):
        l = min(oa) - int(bia)
        if l < 0 :
            l = 0
        r = size + l
    else:
        r = max(oa)+int(bia)
        if r > sizea:
            r = sizea
        l = r - size
    return l,r
def limits(x,y,size,sizea):
    xl,xr = limit(x,size,sizea)
    yl,yr = limit(y,size,sizea) 
    return xl,xr,yl,yr
                
