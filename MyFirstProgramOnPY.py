print("hello world")

if 5 > 2:
  print("Five is greater than two!")

#print("Hello, World!")
print("Cheers, Mate!") # this is a comment too

#this is half comment half string
"""
This is a comment
written in
more than just one line
"""
print("Hello, World!")

x = str(3)    # x will be '3'
y = int(3)    # y will be 3
z = float(3)  # z will be 3.0

print(type(x))
print(type(y))

x = y = z = "Orange"

fruits = ["apple", "banana", "cherry"]
x, y, z = fruits
print(x)
print(y)
print(z)

x = "Python"
y = "is"
z = "awesome"
print(x, y, z)

x = "Python "
y = "is "
z = "awesome"
print(x + y + z)

print(type(x))
print(type(y))
print(type(z))

a = """Lorem ipsum dolor sit amet,
consectetur adipiscing elit,
sed do eiusmod tempor incididunt
ut labore et dolore magna aliqua."""
print(a)

b = "Hello, World!"
print(b[2:5]) #llo
b = "Hello, World!"
print(b[:5])  #Hello
b = "Hello, World!"
print(b[-5:-2]) #orl

a = "Hello, World!"
print(a.lower())

a = "Hello"
b = "World"
c = a + b
print(c)

age = 36
txt = f"My name is John, I am {age}"
print(txt)

price = 59
txt = f"The price is {price:.2f} dollars"
print(txt)

txt = "We are the so-called \"Vikings\" from the north."

x ** y #Степенуване
x // y #Целочислено делене


thislist = ["apple", "banana", "cherry"]
print(thislist)

thislist = ["apple", "banana", "cherry"]
print(thislist[-1]) # cherry

thislist = ["apple", "banana", "cherry"]
thislist[1] = "blackcurrant"
print(thislist)

thislist = ["apple", "banana", "cherry"]
thislist.insert(1, "orange")
print(thislist)



fruits = ["apple", "banana", "cherry", "kiwi", "mango"]
newlist = []

for x in fruits:
  if "a" in x:
    newlist.append(x)

print(newlist)

thislist = ["orange", "mango", "kiwi", "pineapple", "banana"]
thislist.sort(reverse = True)
print(thislist)


mylist = thislist.copy()
print(mylist)

mylist = list(thislist)
print(mylist)