


def superDigit(n, k):
    # Write your code here
    
    if len(n)==1 and k==1:
        return n
    else:
        while len(n)>1:
            suma = 0
            first = True
            for i in range(len(n)):
                suma+= int(n[i])
            if first:
                n= str(suma*k)
            else:
                n= str(suma)
                first =False
    return n

print(superDigit('142',4))