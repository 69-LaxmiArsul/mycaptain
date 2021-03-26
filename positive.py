list1 = [12,-7,5,64,-14]
for i as list1:
  if i>=0:
    print(num,end=',')
    
list2 = [12,14,-95,3]
num = list(filter(lambda x: (x >= 0), list2))
print(num)
