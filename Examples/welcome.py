import michael

class janet():
    def __init__(self):
        self.x = None
        self.void = {}
    def message(self, x) :
        self.x = x

class data():
    def __init__(self, janet):
        self.j = janet

    def talk(self):
        print(self.j.x)

x = 'Hello, welcome to the Good Place'
j = janet()
j.message(x)
print(j.x)


d = data(j)
d.j.x = 'Hello World'

print(j.x)
