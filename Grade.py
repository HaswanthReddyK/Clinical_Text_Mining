a=input("Enter your score")
try:
    b=float(a)
except:
    print("ERROR")
    quit()
if b<=0.6:
    print("F")
elif b==0.6:
    print("D")
elif b==0.7:
    print("C")
elif b==0.8:
    print("B")
elif b==0.9:
    print("A")
else:
    print("Not in range")