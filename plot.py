import matplotlib.pyplot as plt
  
x = []
y = []
y_hat = []
for line in open('output.txt', 'r'):
    lines = [i for i in line.split()]
    x.append(float(lines[0]))
    y.append(float(lines[1]))
    y_hat.append(float(lines[2]))
      
plt.title("Students Marks")
plt.xlabel('x')
plt.ylabel('predict')
plt.scatter(x, y, marker = 'o', c = 'g')

plt.scatter(x, y_hat, marker = 'o', c = 'r')
  
plt.show()