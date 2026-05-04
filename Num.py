smallest=None
largest=None
while True:
    num=input("Enter a number:")
    if num=='done':
        break
    try:
        a = int(num)
    except:
        print("Invalid input")
        continue
    if largest is None:
        largest=a
    elif a>largest:
        largest=a
    if smallest is None:
        smallest=a
    elif a<smallest:
        smallest=a
print("Maximum is",largest)
print("Minimum is",smallest)