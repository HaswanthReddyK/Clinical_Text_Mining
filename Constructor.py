class student:
    def __init__(self,name,section,age,color='red'):
        self.name=name
        self.section=section
        self.age=age
        self.color=color
sectionA=student("Haswanth","A",18)
sectionB=student("Ravi","B",18)
sectionC=student("Manoj","c",18)
print(f"Student from {sectionA.section} SECTION  NAME IS {sectionA.name} got {sectionA.age} his color is {sectionA.color}")
print(f"Student from {sectionB.section} SECTION  NAME IS {sectionB.name} got {sectionB.age}")
print(f"Student from {sectionC.section} SECTION  NAME IS {sectionC.name} got {sectionC.age}")
print(f"Student from {sectionA.section} SECTION  NAME IS {sectionA.name} got {sectionA.age}")



