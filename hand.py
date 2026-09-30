name = input("Enter your name: ")
age = int(input("Enter your age: "))

print("Hello", name)
print("You are", age, "years old.")

num = int(input("Enter a number: "))

if num % 2 == 0:
    print("Even number")
else:
    print("Odd number")
    
num = int(input("Enter a number: "))

for i in range(1, 11):
    print(num, "x", i, "=", num * i)


num = int(input("Enter a number: "))

while num >= 1:
    print(num)
    num -= 1

print("Finished!")


def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    return a / b

a = float(input("Enter first number: "))
b = float(input("Enter second number: "))

print("Addition:", add(a, b))
print("Subtraction:", subtract(a, b))
print("Multiplication:", multiply(a, b))
print("Division:", divide(a, b))

numbers = [10, 25, 7, 45, 18, 32]

largest = numbers[0]

for num in numbers:
    if num > largest:
        largest = num

print("Largest number:", largest)

text = input("Enter a sentence: ")

count = 0

for letter in text:
    if letter.lower() in "aeiou":
        count += 1

print("Number of vowels:", count)

student = {
    "name": "Arish",
    "age": 20,
    "course": "BIT",
    "college": "Kasturi College"
}

print("Name:", student["name"])
print("Age:", student["age"])
print("Course:", student["course"])
print("College:", student["college"])

file = open("student.txt", "w")

file.write("Name: Arish\n")
file.write("Course: BIT\n")
file.write("College: Kasturi College\n")

file.close()

file = open("student.txt", "r")

content = file.read()

print(content)

file.close()

import random

number = random.randint(1, 100)

while True:
    guess = int(input("Guess the number (1-100): "))

    if guess < number:
        print("Too low!")
    elif guess > number:
        print("Too high!")
    else:
        print("Correct! 🎉")
        break