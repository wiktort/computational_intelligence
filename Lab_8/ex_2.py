from simpful import *

# A simple fuzzy inference system for the automatic car breaking problem
# Create a fuzzy system object
FS = FuzzySystem()

# Define fuzzy sets and linguistic variables
D_1 = FuzzySet(function=Trapezoidal_MF(a=0, b=0, c=20, d=45), term="small")
D_2 = FuzzySet(function=Trapezoidal_MF(a=20, b=45, c=105, d=130), term="medium")
D_3 = FuzzySet(function=Trapezoidal_MF(a=105, b=130, c=140, d=140), term="large")

FS.add_linguistic_variable(
    "Distance",
    LinguisticVariable([D_1, D_2, D_3], concept="Distance from lights", universe_of_discourse=[0, 140])
)

S_1 = FuzzySet(function=Trapezoidal_MF(a=0, b=0, c=15, d=65), term="low")
S_2 = FuzzySet(function=Trapezoidal_MF(a=15, b=65, c=65, d=115), term="average")
S_3 = FuzzySet(function=Trapezoidal_MF(a=65, b=115, c=120, d=120), term="high")
FS.add_linguistic_variable(
    "Speed",
    LinguisticVariable([S_1, S_2, S_3], concept="Speed value", universe_of_discourse=[0, 120])
)

# Define output fuzzy sets and linguistic variable
A_1 = FuzzySet(function=Trapezoidal_MF(a=-1, b=-1, c=-0.4, d=0), term="large_m")
A_2 = FuzzySet(function=Trapezoidal_MF(a=-0.4, b=0, c=0, d=0.1), term="small_m")
A_3 = FuzzySet(function=Trapezoidal_MF(a=-0.1, b=0, c=0, d=0.4), term="small_p")
A_4 = FuzzySet(function=Trapezoidal_MF(a=0, b=0.4, c=1, d=1), term="large_p")
FS.add_linguistic_variable("Acceleration", LinguisticVariable([A_1, A_2, A_3, A_4], universe_of_discourse=[-1, 1]))

# Define fuzzy rules
R1 = "IF (Distance IS small) AND (Speed IS low) THEN (Acceleration IS large_m)"
R2 = "IF (Distance IS medium) AND (Speed IS low) THEN (Acceleration IS small_m)"
R3 = "IF (Distance IS large) AND (Speed IS low) THEN (Acceleration IS small_m)"
R4 = "IF (Distance IS small) AND (Speed IS average) THEN (Acceleration IS large_p)"
R5 = "IF (Distance IS medium) AND (Speed IS average) THEN (Acceleration IS small_m)"
R6 = "IF (Distance IS large) AND (Speed IS average) THEN (Acceleration IS small_m)"
R7 = "IF (Distance IS small) AND (Speed IS high) THEN (Acceleration IS small_p)"
R8 = "IF (Distance IS medium) AND (Speed IS high) THEN (Acceleration IS large_m)"
R9 = "IF (Distance IS large) AND (Speed IS high) THEN (Acceleration IS small_m)"

FS.add_rules([R1, R2, R3, R4, R5, R6, R7, R8, R9])

# Set antecedents values and perform Mamdani inference and print output
print("Mamdani\n")
FS.set_variable("Distance", 30)
FS.set_variable("Speed", 50)
print("Distance: 30, Speed: 50 -> {}".format(FS.Mamdani_inference(["Acceleration"])))

FS.set_variable("Distance", 100)
FS.set_variable("Speed", 50)
print("Distance: 100, Speed: 50 -> {}".format(FS.Mamdani_inference(["Acceleration"])))

FS.set_variable("Distance", 30)
FS.set_variable("Speed", 65)
print("Distance: 30, Speed: 65 -> {}".format(FS.Mamdani_inference(["Acceleration"])))
