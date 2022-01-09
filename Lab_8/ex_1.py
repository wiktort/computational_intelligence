from simpful import *

# A simple fuzzy inference system for the tipping problem
# Create a fuzzy system object
FS = FuzzySystem()

# Define fuzzy sets and linguistic variables
S_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=5), term="poor")
S_2 = FuzzySet(function=Triangular_MF(a=0, b=5, c=10), term="good")
S_3 = FuzzySet(function=Triangular_MF(a=5, b=10, c=10), term="excellent")
FS.add_linguistic_variable(
    "Service",
    LinguisticVariable([S_1, S_2, S_3], concept="Service quality", universe_of_discourse=[0, 10])
)

F_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=10), term="rancid")
F_2 = FuzzySet(function=Triangular_MF(a=0, b=10, c=10), term="delicious")
FS.add_linguistic_variable(
    "Food",
    LinguisticVariable([F_1, F_2], concept="Food quality", universe_of_discourse=[0, 10])
)

# Define output fuzzy sets and linguistic variable
T_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=10), term="small")
T_2 = FuzzySet(function=Triangular_MF(a=0, b=10, c=20), term="average")
T_3 = FuzzySet(function=Trapezoidal_MF(a=10, b=20, c=25, d=25), term="generous")
FS.add_linguistic_variable("Tip", LinguisticVariable([T_1, T_2, T_3], universe_of_discourse=[0, 25]))

# Define fuzzy rules
R1 = "IF (Service IS poor) OR (Food IS rancid) THEN (Tip IS small)"
R2 = "IF (Service IS good) THEN (Tip IS average)"
R3 = "IF (Service IS excellent) OR (Food IS delicious) THEN (Tip IS generous)"
FS.add_rules([R1, R2, R3])

# Set antecedents values and perform Mamdani inference and print output
print("Mamdani\n")
FS.set_variable("Service", 4)
FS.set_variable("Food", 8)
print("Service: 4, Food: 8-> {}".format(FS.Mamdani_inference(["Tip"])))

FS.set_variable("Service", 1)
FS.set_variable("Food", 1)
print("Service: 1, Food: 1-> {}".format(FS.Mamdani_inference(["Tip"])))

FS.set_variable("Service", 8)
FS.set_variable("Food", 8)
print("Service: 8, Food: 8-> {}".format(FS.Mamdani_inference(["Tip"])))

FS.set_variable("Service", 10)
FS.set_variable("Food", 10)
print("Service: 10, Food: 10-> {}\n".format(FS.Mamdani_inference(["Tip"])))


# Define output crisp values
FS.set_crisp_output_value("small", 5)
FS.set_crisp_output_value("average", 15)
 # Define function for generous tip (food score + service score + 5%)
FS.set_output_function("generous", "Food+Service+5")

print("Sugeno\n")
FS.set_variable("Service", 4)
FS.set_variable("Food", 8)
print("Service: 4, Food: 8-> {}".format(FS.Sugeno_inference(["Tip"])))

FS.set_variable("Service", 1)
FS.set_variable("Food", 1)
print("Service: 1, Food: 1-> {}".format(FS.Sugeno_inference(["Tip"])))

FS.set_variable("Service", 8)
FS.set_variable("Food", 8)
print("Service: 8, Food: 8-> {}".format(FS.Sugeno_inference(["Tip"])))

FS.set_variable("Service", 10)
FS.set_variable("Food", 10)
print("Service: 10, Food: 10-> {}".format(FS.Sugeno_inference(["Tip"])))

