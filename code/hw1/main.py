import sys
import pickle
import csv
import re

class Person:
    # Initiate person class
    def __init__(self, last, first, mi, id, phone):
        self.last = last
        self.first = first
        self.mi = mi
        self.id = id
        self.phone = phone

    # Display person information
    def display(self):
        print("ID: " + self.id)
        print("Last Name: " + self.last)
        print("First Name: " + self.first)
        print("Middle Initial: " + self.mi)
        print("Phone: " + self.phone)
        print()

# Process CSV data
def process():
    # Initialize useful vars
    map = {}
    id_match = r"^[A-Z]{2}[0-9]{4}$"

    # Open csv
    with open(sys.argv[1]) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        row_count = 0
        # Iterate through each parsed row
        for row in reader:
            print(row)
            if row_count != 0:
                # Check if ID is valid using id matcher
                while re.search(id_match, row[3]) is None:
                    print("ID is invalid: " + row[3])
                    print("ID must be two upper-case letters followed by four numbers.")
                    row[3] = input("Enter new, valid ID: ")

                # parse information
                last = row[0][0].upper() + row[0][1:].lower()
                first = row[1][0].upper() + row[1][1:].lower()
                mi = (row[2].upper(), "X")[row[2] == ""]
                phone = '%s-%s-%s' % tuple(re.findall(r'\d{4}$|\d{3}', row[4]))

                map[row[3]] = Person(last, first, mi, row[3], phone)

            row_count += 1

        return map


def main():
    map = process()

    with open('pickled.pkl', 'wb') as f:
        pickle.dump(map, f)

    with open('pickled.pkl', 'rb') as f:
        unpickled = pickle.load(f)

    for person in unpickled.values():
        person.display()


main()
