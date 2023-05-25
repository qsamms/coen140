file1 = open('output.dat','r')
file2 = open('output2.dat','r')

file1 = []
file2 = []
same = True

for line in file1:
    file1.append(line[0])

for line in file2:
    file2.append(line[0])

for i in range(len(file1)):
    if file1[i] != file2[i]:
       same = False

if same:
    print("They are equal!")
else:
    print("Not Equal!")