a=input("Enter working hours?")
b=input("Enter pay rate per hour")
try:
    c=int(a)
    d=float(b)
except:
    print("Error enter numeric input")
    quit()
if c>40:
    e=c*d
    f=(c-40)*1.5*d
    g=e+f
    print(f"Total pay including overtime${g}")
else:
     e=c*d
     print(f"Total pay{e}")

     #SECOND WAY BELOW


     def computepay(h, r):
         if h > 40:
             regular = 40 * r
             over_time = (h - 40) * 1.5 * r
             d = regular + over_time
         else:
             d = h * r
         return d


     hrs = float(input("Enter Hours:"))
     rate = float(input("enter rate"))
     p = computepay(hrs, rate)
     print("Pay", p)